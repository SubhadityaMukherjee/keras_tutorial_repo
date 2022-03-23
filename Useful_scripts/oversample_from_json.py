# %%
# The data is severly imbalanced and thus favors a single class. This script is to balance it
import json
import pandas as pd
# %%
with open("../web/scripts/fer2013.js") as f:
    js = json.load(f)
# %%
df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in js.items() ]))
#%%
df.shape
# %%
df.head(1)
"""  """
#%%
for i in df.columns:
    print(i, df[i].shape[0])
#%%
import numpy as np
df= df.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
# df.T
#%%
dict_to_s = {x:list(df[x].values) for x in df.columns}
#%%
dict_to_s.keys()
#%%
dict_to_s["angry"]
#%%
# df.T
# df.T.to_json("./web/fer2013.js", orient = "records", lines = True)
# %%
with open("../web/scripts/fer2013.js", "w") as f:
    f.write(json.dumps(dict_to_s, indent = 4))
