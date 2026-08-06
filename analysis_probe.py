"""Defect 1 诊断探针 v3：量化 Predictor 对子图的敏感性。

关键量（在随机初始化 与 3 轮交替训练后 各测一次）：
  1. l_0 (仅 {u,v}) 与 l_t (扩展子图) 的 BCE 之差
  2. 输出 logits 的相对变化 ||pred0 - predt|| / ||pred0||
  3. h_u 在 G_0 与 G_t 中【最终】表征的相对差
  4. pairwise readout 向量的相对差
  5. 首层 GAT 自注意力 / 邻居注意力 比值（用 conv 返回的带自环边索引）
"""

import random
import time

import torch
import torch.nn.functional as F

from model.config import PPIConfig
from model.ppi_model import PPIModel


config = PPIConfig(
    T_max=10,
    gnn_num_layers=3,
    gnn_dropout=0.3,
    gnn_heads=4,
    hidden_dim=256,
    attention_dim=64,
    lr_sampler=1e-4,
    lr_predictor=1e-3,
    sampler_steps=1,
    predictor_steps=1,
    reinforce_baseline_coef=0.5,
)


def subgraph_h(model, esm, graph, sub, core, want_attn=False, edges=None):
    """复现 predictor 前向，返回 (h_final, pairwise, conv1_attn)。"""
    x, ei, ul, vl = model.predictor._build_subgraph(
        esm, graph, sub, core, edges=edges)
    h = model.predictor.node_proj(x)
    conv0 = model.predictor.convs[0]
    if want_attn:
        conv0.return_attention_weights = True
        _, (ei_loop, att) = conv0(h, ei, return_attention_weights=True)
        conv0.return_attention_weights = False
    else:
        conv0.return_attention_weights = False
        h_new = conv0(h, ei)
        h_new = model.predictor.norms[0](h_new)
        h_new = F.relu(h_new)
        h_new = F.dropout(h_new, p=model.config.gnn_dropout, training=False)
        h = h + h_new
        for conv, norm in zip(model.predictor.convs[1:], model.predictor.norms[1:]):
            h_new = conv(h, ei)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=model.config.gnn_dropout, training=False)
            h = h + h_new
        ei_loop, att = None, None

    if want_attn:
        # 后续层继续传播（仅用于拿完整 h，注意力只看首层）
        for conv, norm in zip(model.predictor.convs[1:], model.predictor.norms[1:]):
            h_new = conv(h, ei)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=model.config.gnn_dropout, training=False)
            h = h + h_new

    hu, hv = h[ul], h[vl]
    pairwise = torch.cat([hu, hv, hu * hv, torch.abs(hu - hv)], dim=-1)
    return h, pairwise, (ei_loop, att)


def main():
    print("[probe] building model ...")
    model = PPIModel.from_dataset(config, "SHS27k", "dataset", verbose=False)
    model.set_esm_tensor("dataset/SHS27k_tensor.pt")
    split = model.load_split("dataset/SHS27k_bfs.json")
    train_idx = split["train_index"]

    rng = random.Random(0)
    probe_ppi = rng.sample(train_idx, 60)
    esm = model._esm_tensor.float().to(model.config.device)

    def probe_state(tag):
        model.sampler.eval()
        model.predictor.eval()
        l0s, lts = [], []
        d_logit, d_hu_final, d_pairwise = [], [], []
        self_att, nbr_att = [], []

        for ppi in probe_ppi:
            u, v = model.ppi_list[ppi]
            label = model._get_label(ppi).to(model.config.device)

            # --- G_0 = {u, v} ---
            pred0 = model.predictor(model._esm_tensor, model.graph, [u, v], (u, v))
            l0 = F.binary_cross_entropy(pred0, label)
            h0, pw0, _ = subgraph_h(model, esm, model.graph, [u, v], (u, v))

            # --- G_t = sampler argmax 扩展子图（含诱导边） ---
            traj = model.sampler(model._esm_tensor, model.graph, u, v,
                                 training=False)
            sub, sub_edges = traj.final_subgraph, traj.final_edges
            predt = model.predictor(model._esm_tensor, model.graph, sub,
                                    (u, v), edges=sub_edges)
            lt = F.binary_cross_entropy(predt, label)
            ht, pwt, (ei_loop, att) = subgraph_h(
                model, esm, model.graph, sub, (u, v),
                want_attn=True, edges=sub_edges)

            l0s.append(l0.item())
            lts.append(lt.item())
            d_logit.append((pred0 - predt).norm().item() / pred0.norm().item())
            d_hu_final.append((h0[0] - ht[0]).norm().item() / h0[0].norm().item())
            d_pairwise.append((pw0 - pwt).norm().item() / pw0.norm().item())

            # --- 首层 GAT 注意力：自环 vs 邻居 ---
            att = att.mean(dim=-1)
            rows, cols = ei_loop
            is_self = (rows == cols)
            if is_self.any() and (~is_self).any():
                self_att.append(att[is_self].mean().item())
                nbr_att.append(att[~is_self].mean().item())

        n = len(probe_ppi)
        print(f"\n===== {tag} =====")
        print(f"  l_0  mean={sum(l0s)/n:.4f}")
        print(f"  l_t  mean={sum(lts)/n:.4f}")
        print(f"  l_0 - l_t (reward) mean={sum(a-b for a,b in zip(l0s,lts))/n:+.5f}")
        print(f"  ||Δlogits|| / ||logits||     mean={sum(d_logit)/n:.4f}")
        print(f"  ||Δh_u^final|| / ||h_u(G0)|| mean={sum(d_hu_final)/n:.4f}")
        print(f"  ||Δpairwise|| / ||pairwise|| mean={sum(d_pairwise)/n:.4f}")
        if self_att:
            print(f"  首层GAT 自注意力={sum(self_att)/len(self_att):.4f} | "
                  f"邻居注意力={sum(nbr_att)/len(nbr_att):.4f} | "
                  f"自/邻={sum(s/n for s,n in zip(self_att,nbr_att))/len(self_att):.2f}")

    probe_state("随机初始化（未训练）")

    print("\n[probe] training 3 rounds ...")
    for rnd in range(1, 4):
        t0 = time.time()
        pool_cur = rng.sample(train_idx, 600)
        model.train_sampler_step(pool_cur[:600])
        model.train_predictor_step(pool_cur[:600])
        print(f"  round {rnd} done in {time.time()-t0:.1f}s")

    probe_state("3 轮交替训练后")


if __name__ == "__main__":
    main()
