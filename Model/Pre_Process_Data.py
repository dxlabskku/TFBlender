import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

############################
# 0) Group mapping CSV loader
############################
def load_fred_group(path):
    df = pd.read_csv(path, encoding='cp949')
    group_map = dict(zip(df.iloc[:,0], df.iloc[:,1]))
    return group_map

############################
# 1) Weekly re-sampling (e.g., 128 or 16 weeks)
############################
def resample_to_fixed_weekly(data, issue_date, target_len=128):
    """
    Resamples price data (or similar) into fixed weekly intervals going backward from 'issue_date'.
    """
    data = data.set_index('sasdate')
    weekly_data = []
    current_date = issue_date

    while len(weekly_data) < target_len and current_date >= data.index.min():
        w_slice = data.loc[current_date - pd.Timedelta(days=6): current_date]
        w_mean = w_slice.mean()
        w_mean.name = current_date
        weekly_data.append(w_mean)
        current_date -= pd.Timedelta(weeks=1)

    weekly_data = pd.DataFrame(weekly_data).reset_index()
    weekly_data.columns = ['sasdate'] + list(data.columns)
    weekly_data = weekly_data.sort_values(by='sasdate').reset_index(drop=True)
    return weekly_data

############################
# 2) Extend macro data in weekly units
############################
def extend_to_fixed_length(data, weekly_price_data):
    """
    Merges the weekly price data index with macro data, matching rows on the most recent available date.
    This effectively expands data from monthly or quarterly frequency to weekly frequency.
    """
    extended_data = pd.merge_asof(
        weekly_price_data[['sasdate']],
        data.sort_values('sasdate').reset_index(drop=True),
        on='sasdate',
        direction='backward'
    )
    return extended_data.drop(columns=['sasdate']).reset_index(drop=True)

############################
# 3) (Method B) Build pyramids by group + remove duplicate columns
############################
def create_pyramid_data_by_group(
    els_data_scaled,
    price_data_scaled,
    month_data_scaled,
    quarter_data_scaled,
    group_map,
    target_len=64
):
    """
    1) price_data_scaled : Scaled price data (includes 'sasdate')
    2) month_data_scaled : Scaled monthly data (includes 'sasdate')
    3) quarter_data_scaled : Scaled quarterly data (includes 'sasdate')
    4) group_map : Mapping of {indicator_name: group_number} from fred_group.csv
    """
    all_pyramid_data = []
    prev_issue_date = None
    prev_pyramid_data = None

    for issue_date in els_data_scaled['Issue_Date']:
        if issue_date == prev_issue_date:
            # If the issue date is the same, reuse the previous result
            all_pyramid_data.append(prev_pyramid_data)
        else:
            # A) Resample stock price data into the specified weekly length
            weekly_price_data = resample_to_fixed_weekly(price_data_scaled, issue_date, target_len)

            # B) Extend monthly data in weekly intervals
            extended_month = extend_to_fixed_length(month_data_scaled, weekly_price_data)

            # C) Extend quarterly data in weekly intervals
            extended_quarter = extend_to_fixed_length(quarter_data_scaled, weekly_price_data)

            # D) Remove duplicate columns: if monthly and quarterly share the same header, remove from quarterly
            common_cols = set(extended_month.columns).intersection(set(extended_quarter.columns))
            if common_cols:
                extended_quarter = extended_quarter.drop(columns=list(common_cols), errors='ignore')

            # E) Final macro data = monthly + quarterly concatenation
            combined_mq = pd.concat([extended_month, extended_quarter], axis=1)

            # F) Apply group mapping
            pyramid_item = {}
            # Price data
            pyramid_item['price'] = weekly_price_data.drop(columns=['sasdate'])

            # Gather columns by group
            group_to_cols = {}
            for col in combined_mq.columns:
                if col in group_map: 
                    grp = str(group_map[col])
                    if grp not in group_to_cols:
                        group_to_cols[grp] = []
                    group_to_cols[grp].append(col)

            # Create group_{g}
            for g, cols in group_to_cols.items():
                pyramid_item[f'group_{g}'] = combined_mq[cols]

            all_pyramid_data.append(pyramid_item)
            prev_issue_date = issue_date
            prev_pyramid_data = pyramid_item

    return all_pyramid_data

############################
# 4) Periodic encoding for issue month/date
############################
def encode_time_features(df):
    df['Issue_Month_sin'] = np.sin(2 * np.pi * df['Issue_Month'] / 12)
    df['Issue_Month_cos'] = np.cos(2 * np.pi * df['Issue_Month'] / 12)
    df['Issue_Day_sin'] = np.sin(2 * np.pi * df['Issue_Day'] / 31)
    df['Issue_Day_cos'] = np.cos(2 * np.pi * df['Issue_Day'] / 31)
    return df

############################
# 5) Main execution (Full Method B pipeline)
############################
if __name__ == "__main__":
    # (A) Read CSV files
    price_data   = pd.read_csv("stock_from_1998.csv",  encoding='cp949')
    els_data     = pd.read_csv("ELS_data.csv",         encoding='cp949')
    month_data   = pd.read_csv("fred_month.csv",       encoding='cp949')
    quarter_data = pd.read_csv("fred_quarter.csv",     encoding='cp949')
    group_map    = load_fred_group("fred_group.csv")

    # (B) Convert date columns
    price_data['sasdate']   = pd.to_datetime(price_data['sasdate'])
    els_data['Issue_Date']      = pd.to_datetime(els_data['Issue_Date'])
    month_data['sasdate']   = pd.to_datetime(month_data['sasdate'])
    quarter_data['sasdate'] = pd.to_datetime(quarter_data['sasdate'])

    # (C) Scaling
    scaler = StandardScaler()

    # Price
    price_no_date = price_data.drop(columns=['sasdate'])
    price_scaled_vals = scaler.fit_transform(price_no_date)
    price_data_scaled = pd.DataFrame(price_scaled_vals, columns=price_no_date.columns)
    price_data_scaled = pd.concat([price_data_scaled, price_data[['sasdate']].reset_index(drop=True)], axis=1)

    # Monthly
    month_no_date = month_data.drop(columns=['sasdate'])
    month_scaled_vals = scaler.fit_transform(month_no_date)
    month_data_scaled = pd.DataFrame(month_scaled_vals, columns=month_no_date.columns)
    month_data_scaled = pd.concat([month_data_scaled, month_data[['sasdate']].reset_index(drop=True)], axis=1)

    # Quarterly
    quarter_no_date = quarter_data.drop(columns=['sasdate'])
    quarter_scaled_vals = scaler.fit_transform(quarter_no_date)
    quarter_data_scaled = pd.DataFrame(quarter_scaled_vals, columns=quarter_no_date.columns)
    quarter_data_scaled = pd.concat([quarter_data_scaled, quarter_data[['sasdate']].reset_index(drop=True)], axis=1)

    # (D) Binary label
    profit_result = els_data['Knock_In_Barrior'].values
    binary_labels = [1 if label == 'Yes' else 0 for label in profit_result]

    # (E) Time encoding for ELS
    els_data = encode_time_features(els_data)

    # Drop unused columns
    drop_cols_els = [
        'Knock_In_Asset','Knock_In_Date','Deter_Date','Issue_Month_sin','Issue_Month_cos',
        'Issue_Day_sin','Issue_Day_cos','Issue_Date','Expire_Date','Issue_Month',
        'Issue_Day','Knock_In_Barrior','Underlying_Asset1','Underlying_Asset2','Underlying_Asset3',
        'Underlying_Asset_Price1','Underlying_Asset_Price2','Underlying_Asset_Price3'
    ]
    els_reduced = els_data.drop(columns=drop_cols_els, errors='ignore')
    els_scaled_vals = scaler.fit_transform(els_reduced)
    els_data_scaled = pd.DataFrame(els_scaled_vals, columns=els_reduced.columns)

    # Re-attach columns that were not scaled
    reattach_cols = els_data[[
        'Deter_Date','Issue_Date','Issue_Month_sin','Issue_Month_cos',
        'Issue_Day_sin','Issue_Day_cos','Underlying_Asset1','Underlying_Asset2',
        'Underlying_Asset3','Underlying_Asset_Price 대비','Underlying_Asset_Price1','Underlying_Asset_Price2','Underlying_Asset_Price3'
    ]].reset_index(drop=True)
    els_data_scaled = pd.concat([reattach_cols, els_data_scaled.reset_index(drop=True)], axis=1)

    # (F) Pyramid creation by group (Method B + remove duplicate columns)
    all_pyramid_data = create_pyramid_data_by_group(
        els_data_scaled     = els_data_scaled,
        price_data_scaled   = price_data_scaled,
        month_data_scaled   = month_data_scaled,
        quarter_data_scaled = quarter_data_scaled,
        group_map           = group_map,
        target_len          = 16
    )

    # (G) One-hot encode the underlying assets + weighting by reference price
    asset_cols = ['Underlying_Asset1','Underlying_Asset2','Underlying_Asset3']
    asset_val_cols = ['Underlying_Asset_Price1','Underlying_Asset_Price2','Underlying_Asset_Price3']
    encoder_assets = OneHotEncoder(sparse_output=False)
    encoded_assets = encoder_assets.fit_transform(els_data_scaled[asset_cols])
    encoded_assets_df = pd.DataFrame(encoded_assets, columns=encoder_assets.get_feature_names_out(asset_cols))

    # Multiply one-hot results by each reference price
    encoded_assets_1 = encoded_assets_df.filter(like='Underlying_Asset1').mul(els_data_scaled['Underlying_Asset_Price1'], axis=0)
    encoded_assets_2 = encoded_assets_df.filter(like='Underlying_Asset2').mul(els_data_scaled['Underlying_Asset_Price2'], axis=0)
    encoded_assets_3 = encoded_assets_df.filter(like='Underlying_Asset3').mul(els_data_scaled['Underlying_Asset_Price3'], axis=0)

    # Drop these columns from the main DataFrame
    els_data_scaled = els_data_scaled.drop(columns=asset_cols + asset_val_cols, errors='ignore')
    weighted_assets = pd.concat([encoded_assets_1, encoded_assets_2, encoded_assets_3], axis=1)
    weighted_assets = weighted_assets.drop(columns=weighted_assets.filter(like='_0').columns)

    # Scale the weighted assets
    weighted_assets_scaled = pd.DataFrame(scaler.fit_transform(weighted_assets), columns=weighted_assets.columns)

    # Combine everything into one final DataFrame
    final_df = pd.concat([
        els_data_scaled.reset_index(drop=True),
        weighted_assets_scaled.reset_index(drop=True)
    ], axis=1)

    # Convert timestamps to numeric
    final_df['Issue_Date'] = pd.to_datetime(final_df['Issue_Date'], errors='coerce').apply(
        lambda x: x.timestamp() if pd.notnull(x) else 0
    )
    final_df['Deter_Date'] = pd.to_datetime(final_df['Deter_Date'], errors='coerce').apply(
        lambda x: x.timestamp() if pd.notnull(x) else 0
    )

    # Create tensors
    binary_labels_tensor = torch.tensor(binary_labels, dtype=torch.float32)
    x_condition_tensor_final = torch.tensor(final_df.values, dtype=torch.float32)

    # Convert all pyramid data to tensors
    all_pyramid_data_tensor = []
    for data_item in all_pyramid_data:
        price_tensor = torch.tensor(data_item['price'].values, dtype=torch.float32)
        group_tensors = {}
        for k,v in data_item.items():
            if k == 'price':
                continue
            group_tensors[k] = torch.tensor(v.values, dtype=torch.float32)
        pyramid_dict = {'price': price_tensor}
        pyramid_dict.update(group_tensors)
        all_pyramid_data_tensor.append(pyramid_dict)
