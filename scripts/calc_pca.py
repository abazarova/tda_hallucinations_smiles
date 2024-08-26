import json

import numpy as np
from sklearn.decomposition import PCA
from tqdm import trange

if __name__=="__main__":
    with open("/app/assets/attention_maps/input_ids_llama-2-7b-chat.json", "r") as f:
        input_ids_total = json.load(f)

    with open("/app/assets/attention_maps/prompt_lengths.json", "r") as f:
        ntokens_prompt = json.load(f)

    for i in trange(6):
        point_clouds = []
        matrices = np.load(f"/app/assets/attention_maps/llama-2-7b-chat/pt_{i}/attn_matrices.npz")
        with open(f"/app/assets/attention_maps/llama-2-7b-chat/pt_{i}/tokens_count.json", "r") as f:
            tokens_count = json.load(f)

        n_layers, n_heads = 2, 32
        for key in list(matrices.keys()):
            ntokens = tokens_count[key]
            ntokens_prt = ntokens_prompt[key]
            mxs = matrices[key][:, :, :ntokens, :ntokens]
            point_cloud = []
            for l in range(n_layers):
                for j in range(n_heads):
                    pca = PCA(n_components=5)
                    mx = mxs[l][j]
                    projection = pca.fit_transform(mx)
                    point_cloud.append(projection)
                    
            point_clouds.append(np.stack(point_cloud))

        np.save(f"point_clouds_{i}_pca", point_clouds)