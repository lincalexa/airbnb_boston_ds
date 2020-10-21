import pandas as pd
import numpy as np


def drop_columns_missing_data(df, missing_threshhold):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    missing_threshhold - a decimal number representing the cutoff threshold for the missing data.  
        Ex 0.75 will drop columns with > 75% missing data
    
    OUTPUT:
    df - a dataframe with low data columns removed
    '''
    # drop columns with little data
    most_missing_cols = np.array(df.columns[df.isnull().sum(axis=0) >= (df.shape[0] * missing_threshhold)])

    # drop these as they don't add value
    df = df.drop(columns=most_missing_cols)
    return df

def convert_dollar_to_float(df, dollar_cols):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    dollar_cols - list of string type dollar columns to convert
    
    OUTPUT:
    df - a dataframe with string dollars converted to floats
    '''
    # start by making dollar columns a float instead of string
    df[dollar_cols] = df[dollar_cols].replace('[\$,]', '', regex=True).astype(float)
    return df


def reencode_multilevel_features(df):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    
    OUTPUT:
    df - a dataframe with multi-level features re-encoded
    '''
    df = pd.get_dummies(df, columns=df.select_dtypes('object').columns)
    return df


def clean_listings_data(df, calendar, with_encoding=True):
    '''
    INPUT:
    df - a dataframe with listing data to be cleaned
    calendar - a dataframe with calendar data to be cleaned
    
    OUTPUT:
    df - a cleaned up dataframe ready for analysis
    '''    
    # drop any row with more than 75% missing data
    df = drop_columns_missing_data(df, 0.75)
    
    # drop these columns as they are very verbose, or not interesting for the current project, or data not varied
    df = df.drop(columns=['listing_url', 'last_scraped','experiences_offered','space',
       'description', 'neighborhood_overview', 'notes','transit','access','interaction','house_rules',
       'host_about','host_response_time','host_thumbnail_url', 'host_picture_url','neighbourhood','market',
      'state','city','smart_location', 'country_code', 'country','is_location_exact','amenities','calendar_last_scraped',
      'first_review','last_review','requires_license','instant_bookable','require_guest_profile_picture',
      'require_guest_phone_verification','host_name','host_location','thumbnail_url', 'medium_url', 'picture_url',
       'xl_picture_url', 'host_url','host_neighbourhood','zipcode','host_has_profile_pic','street',
      'host_response_rate','scrape_id', 'summary', 'host_verifications', 'latitude', 'longitude',
      'calendar_updated', 'host_acceptance_rate', 'name', 'host_since','availability_30',
      'availability_60', 'availability_90', 'availability_365', 
      'calculated_host_listings_count','host_total_listings_count'])
    
    # start by making dollar columns a float instead of string
    dollar_cols = ['price','cleaning_fee', 'extra_people', 'security_deposit']
    df = convert_dollar_to_float(df, dollar_cols)
    
    # fill in the missing bed and bath data
    # for the 14 NaN bathrooms, set to 1
    # for the 10 NaN bedrooms, set to 1
    # for the 9 NaN beds, set to 1
    fill_one = lambda col: col.fillna(1)
    df[['bathrooms','bedrooms','beds']] = df[['bathrooms','bedrooms','beds']].apply(fill_one, axis=0)
    
    # 2243 listings have NaN for security_deposit - set to zero or drop
    fill_zero = lambda col: col.fillna(0)
    df[['security_deposit']] = df[['security_deposit']].apply(fill_zero, axis=0)

    # 1107 listings have NaN for cleaning_fee - consider setting to mean - 68.38014527845036
    fill_mean = lambda col: col.fillna(col.mean())
    df[['cleaning_fee']] = df[['cleaning_fee']].apply(fill_mean, axis=0)

    # fill in the missing property_type data with 'Apartment'
    df[['property_type']] = df[['property_type']].apply(lambda col: col.fillna('Apartment'), axis=0)
    
    # for questions around reviews - need to drop the rows that are have missing review data
    ratings_with_nulls = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
                          'review_scores_checkin','review_scores_communication', 'review_scores_location',
                          'review_scores_value','reviews_per_month']

    for na_col in ratings_with_nulls:
        df = df[df[na_col].notna()]
        
    if with_encoding:
        # use get_dummies to re-encode multi-level features
        df = reencode_multilevel_features(df)
        
    # feature engineering
    calendar_clean = clean_calendar(calendar.copy())
    calendar_monthly_avgs = get_monthly_avg_rentals(calendar_clean)
    set_avg_monthly_days_rented(df, calendar_monthly_avgs)

    return df

# for each listing_id = add columns for each to listings dataset - 
# avg_monthly_days_rented_2016 and avg_monthly_days_rented_2017
def set_avg_monthly_days_rented(df_listing, df_avg_cal):
    '''
    INPUT:
    df_listing - DataFrame of listings data
    df_avg_cal - DataFrame of calendar data engineered to add average monthly rental features
    
    OUTPUT:
    df_listing - DataFrame of listing data with the calendar data added
    '''    
    df_listing['avg_monthly_days_rented_2016'] = 0
    df_listing['avg_monthly_days_rented_2017'] = 0

    for index, row in df_listing.iterrows():
        # use row['id'] to query avail_avg_monthly to get month and year
        avg_rows = df_avg_cal[df_avg_cal['listing_id'] == row['id']]
        for index2, row2 in avg_rows.iterrows():
            if row2['year'] == 2016:
                df_listing.at[index,'avg_monthly_days_rented_2016'] = row2['avg_monthly_days_rented']
            else:
                df_listing.at[index,'avg_monthly_days_rented_2017'] = row2['avg_monthly_days_rented']

    return df_listing


# get_monthly average rentals
def get_monthly_avg_rentals(df):
    '''
    INPUT:
    df - a DataFrame with calendar data for feature engineering
    
    OUTPUT:
    df - DataFrame of average monthy days rented counts per listing per year
    '''    
    # just get values for dates unavailable as those indicate it was rented
    df_unavail = df[df['available'] == 'f']
    # get counts by month
    unavail_group = df_unavail.groupby(['listing_id','year','month']).available.count().to_frame(name='month_count').reset_index()
    # get monthly averages
    unavail_avg_monthly = unavail_group.groupby(['listing_id','year']).month_count.mean().to_frame(name='avg_monthly_days_rented').reset_index()
    return unavail_avg_monthly


def clean_calendar(df):
    '''
    INPUT:
    df - a dataframe with calendar data to be cleaned
    
    OUTPUT:
    df - a cleaned up dataframe ready for analysis
    '''    
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df