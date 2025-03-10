import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

def load_fred_group(path):
    df = pd.read_csv(path, encoding='cp949')
    return dict(zip(df.iloc[:,0], df.iloc[:,1]))

def resample_to_fixed_weekly(data, issue_date, target_len=128):
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

def extend_to_fixed_length(data, weekly_price_data):
    extended_data = pd.merge_asof(
        weekly_price_data[['sasdate']],
        data.sort_values('sasdate').reset_index(drop=True),
        on='sasdate',
        direction='backward'
    )
    return extended_data.drop(columns=['sasdate']).reset_index(drop=True)

def create_frames_by_group(
    els_data_scaled,
    price_data_scaled,
    month_data_scaled,
    quarter_data_scaled,
    group_map,
    target_len=64
):
    all_frame_data = []
    prev_issue_date = None
    prev_frame = None

    for issue_date in els_data_scaled['Issue_Date']:
        if issue_date == prev_issue_date:
            all_frame_data.append(prev_frame)
        else:
            weekly_price_data = resample_to_fixed_weekly(price_data_scaled, issue_date, target_len)
            extended_month = extend_to_fixed_length(month_data_scaled, weekly_price_data)
            extended_quarter = extend_to_fixed_length(quarter_data_scaled, weekly_price_data)
            common_cols = set(extended_month.columns).intersection(set(extended_quarter.columns))
            if common_cols:
                extended_quarter = extended_quarter.drop(columns=list(common_cols), errors='ignore')
            combined_mq = pd.concat([extended_month, extended_quarter], axis=1)

            frame_item = weekly_price_data.drop(columns=['sasdate']).copy()
            group_to_cols = {}
            for col in combined_mq.columns:
                if col in group_map:
                    grp = str(group_map[col])
                    if grp not in group_to_cols:
                        group_to_cols[grp] = []
                    group_to_cols[grp].append(col)

            for g, cols in group_to_cols.items():
                sub_df = combined_mq[cols].add_prefix(f'group_{g}_')
                frame_item = pd.concat([frame_item, sub_df], axis=1)

            all_frame_data.append(frame_item)
            prev_issue_date = issue_date
            prev_frame = frame_item
    return all_frame_data

def encode_time_features(df):
    df['Issue_Month_sin'] = np.sin(2 * np.pi * df['Issue_Month'] / 12)
    df['Issue_Month_cos'] = np.cos(2 * np.pi * df['Issue_Month'] / 12)
    df['Issue_Day_sin'] = np.sin(2 * np.pi * df['Issue_Day'] / 31)
    df['Issue_Day_cos'] = np.cos(2 * np.pi * df['Issue_Day'] / 31)
    return df

if __name__ == "__main__":
    price_data = pd.read_csv("stock_from_1998.csv", encoding='cp949') # Need to collect from yfinance
    els_data = pd.read_csv("ELS_terms.csv", encoding='cp949')
    month_data = pd.read_csv("fred_month.csv", encoding='cp949')
    quarter_data = pd.read_csv("fred_quarter.csv", encoding='cp949')
    group_map = load_fred_group("fred_group.csv")

    price_data['sasdate'] = pd.to_datetime(price_data['sasdate'])
    els_data['Issue_Date'] = pd.to_datetime(els_data['Issue_Date'])
    month_data['sasdate'] = pd.to_datetime(month_data['sasdate'])
    quarter_data['sasdate'] = pd.to_datetime(quarter_data['sasdate'])

    scaler = StandardScaler()
    price_no_date = price_data.drop(columns=['sasdate'])
    price_scaled_vals = scaler.fit_transform(price_no_date)
    price_data_scaled = pd.DataFrame(price_scaled_vals, columns=price_no_date.columns)
    price_data_scaled = pd.concat([price_data_scaled, price_data[['sasdate']].reset_index(drop=True)], axis=1)

    month_no_date = month_data.drop(columns=['sasdate'])
    month_scaled_vals = scaler.fit_transform(month_no_date)
    month_data_scaled = pd.DataFrame(month_scaled_vals, columns=month_no_date.columns)
    month_data_scaled = pd.concat([month_data_scaled, month_data[['sasdate']].reset_index(drop=True)], axis=1)

    quarter_no_date = quarter_data.drop(columns=['sasdate'])
    quarter_scaled_vals = scaler.fit_transform(quarter_no_date)
    quarter_data_scaled = pd.DataFrame(quarter_scaled_vals, columns=quarter_no_date.columns)
    quarter_data_scaled = pd.concat([quarter_data_scaled, quarter_data[['sasdate']].reset_index(drop=True)], axis=1)

    profit_result = els_data['Knock_In_Barrior'].values
    binary_labels = [1 if label == 'Yes' else 0 for label in profit_result]

    els_data = encode_time_features(els_data)
    drop_cols_els = [
        'Knock_In_Asset','Knock_In_Date','Deter_Date','Issue_Month_sin','Issue_Month_cos',
        'Issue_Day_sin','Issue_Day_cos','Issue_Date','Expire_Date','Issue_Month',
        'Issue_Day','Knock_In_Barrior','Underlying_Asset1','Underlying_Asset2','Underlying_Asset3',
        'Underlying_Asset_Price1','Underlying_Asset_Price2','Underlying_Asset_Price3'
    ]
    els_reduced = els_data.drop(columns=drop_cols_els, errors='ignore')
    els_scaled_vals = scaler.fit_transform(els_reduced)
    els_data_scaled = pd.DataFrame(els_scaled_vals, columns=els_reduced.columns)
    reattach_cols = els_data[[
        'Deter_Date','Issue_Date','Issue_Month_sin','Issue_Month_cos',
        'Issue_Day_sin','Issue_Day_cos','Underlying_Asset1','Underlying_Asset2',
        'Underlying_Asset3','Knock_In_Barrior','Underlying_Asset_Price1','Underlying_Asset_Price2','Underlying_Asset_Price3'
    ]].reset_index(drop=True)
    els_data_scaled = pd.concat([reattach_cols, els_data_scaled.reset_index(drop=True)], axis=1)

    all_frame_data = create_frames_by_group(
        els_data_scaled=els_data_scaled,
        price_data_scaled=price_data_scaled,
        month_data_scaled=month_data_scaled,
        quarter_data_scaled=quarter_data_scaled,
        group_map=group_map,
        target_len=16
    )

    asset_cols = ['Underlying_Asset1','Underlying_Asset2','Underlying_Asset3']
    asset_val_cols = ['Underlying_Asset_Price1','Underlying_Asset_Price2','Underlying_Asset_Price3']
  
    encoder_assets = OneHotEncoder(sparse_output=False)
    encoded_assets = encoder_assets.fit_transform(els_data_scaled[asset_cols])
    encoded_assets_df = pd.DataFrame(encoded_assets, columns=encoder_assets.get_feature_names_out(asset_cols))
  
    encoded_assets_1 = encoded_assets_df.filter(like='Underlying_Asset1').mul(els_data_scaled['Underlying_Asset_Price1'], axis=0)
    encoded_assets_2 = encoded_assets_df.filter(like='Underlying_Asset2').mul(els_data_scaled['Underlying_Asset_Price2'], axis=0)
    encoded_assets_3 = encoded_assets_df.filter(like='Underlying_Asset3').mul(els_data_scaled['Underlying_Asset_Price3'], axis=0)

    els_data_scaled = els_data_scaled.drop(columns=asset_cols + asset_val_cols, errors='ignore')
    weighted_assets = pd.concat([encoded_assets_1, encoded_assets_2, encoded_assets_3], axis=1)
    weighted_assets = weighted_assets.drop(columns=weighted_assets.filter(like='_0').columns)
    weighted_assets_scaled = pd.DataFrame(scaler.fit_transform(weighted_assets), columns=weighted_assets.columns)

    final_df = pd.concat([
        els_data_scaled.reset_index(drop=True),
        weighted_assets_scaled.reset_index(drop=True)
    ], axis=1)

    final_df['Issue_Date'] = pd.to_datetime(final_df['Issue_Date'], errors='coerce').apply(
        lambda x: x.timestamp() if pd.notnull(x) else 0
    )
    final_df['Deter_Date'] = pd.to_datetime(final_df['Deter_Date'], errors='coerce').apply(
        lambda x: x.timestamp() if pd.notnull(x) else 0
    )

    binary_labels_tensor = torch.tensor(binary_labels, dtype=torch.float32)
    x_condition_tensor_final = torch.tensor(final_df.values, dtype=torch.float32)
