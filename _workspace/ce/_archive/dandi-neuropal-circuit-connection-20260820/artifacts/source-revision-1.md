# Source revision 1

Status: COMPLETE

The preregistered DANDI 000565 `<300 MB` selection failed the schema gate before any response endpoint was computed. The downloaded asset contained a static six-channel NeuroPAL volume, 116 ROI rows, ID labels and voxel masks, but no calcium time series and an empty stimulus presentation group. The failure is retained in `first-asset-schema.json`; `source-manifest.json` is superseded but not deleted.

The replacement source is DANDI 000541 version `0.241009.1457`, which the official dataset and associated paper describe as head NeuroPAL and calcium imaging at about 4 Hz for about four minutes. The eight smallest published assets were frozen by official content size before any 000541 response array was opened. Exact metadata are in `source-manifest-v2.json`.

Because 000541 does not advertise an optogenetic assignment receipt, the revised route is observational and uses a time-reversal contrast. No causal-parent or intervention claim is permitted.

