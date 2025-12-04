# Feature Transformation Table
Automatically generated after running the feature pipeline.

| Feature | Rule | Output Columns |
|---------|------|----------------|
| `PPH_Open` | `no_shift` | `PPH_Open` |
| `PPH_High` | `no_shift` | `PPH_High` |
| `PPH_Low` | `no_shift` | `PPH_Low` |
| `PPH_Close` | `no_shift` | `PPH_Close` |
| `PPH_Volume` | `no_shift` | `PPH_Volume` |
| `XPH_Open` | `no_shift` | `XPH_Open` |
| `XPH_High` | `no_shift` | `XPH_High` |
| `XPH_Low` | `no_shift` | `XPH_Low` |
| `XPH_Close` | `no_shift` | `XPH_Close` |
| `XPH_Volume` | `no_shift` | `XPH_Volume` |
| `target_close` | `no_shift` | `target_close` |
| `PPH_Return_1d` | `shift_1` | `PPH_Return_1d_t-1` |
| `PPH_SMA_10` | `shift_1` | `PPH_SMA_10_t-1` |
| `PPH_RSI_14` | `shift_1` | `PPH_RSI_14_t-1` |
| `PPH_StochK` | `shift_1` | `PPH_StochK_t-1` |
| `PPH_StochD` | `shift_1` | `PPH_StochD_t-1` |
| `PPH_Entropy_20` | `shift_1` | `PPH_Entropy_20_t-1` |
| `PPH_HMM` | `shift_1` | `PPH_HMM_t-1` |
| `XPH_Return_1d` | `shift_1` | `XPH_Return_1d_t-1` |
| `XPH_SMA_10` | `shift_1` | `XPH_SMA_10_t-1` |
| `XPH_RSI_14` | `shift_1` | `XPH_RSI_14_t-1` |
| `XPH_StochK` | `shift_1` | `XPH_StochK_t-1` |
| `XPH_StochD` | `shift_1` | `XPH_StochD_t-1` |
| `XPH_Entropy_20` | `shift_1` | `XPH_Entropy_20_t-1` |
| `XPH_HMM` | `shift_1` | `XPH_HMM_t-1` |
| `XPH_Ratio_PPH` | `shift_1` | `XPH_Ratio_PPH_t-1` |
| `day_of_week` | `no_shift` | `day_of_week` |
| `day_of_month` | `no_shift` | `day_of_month` |
| `month` | `no_shift` | `month` |
| `quarter` | `no_shift` | `quarter` |
| `is_holiday_adjacent` | `no_shift` | `is_holiday_adjacent` |
| `days_to_cpi` | `no_shift` | `days_to_cpi` |
| `days_since_cpi` | `no_shift` | `days_since_cpi` |
| `days_to_nfp` | `no_shift` | `days_to_nfp` |
| `days_since_nfp` | `no_shift` | `days_since_nfp` |
