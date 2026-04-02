# #!/usr/bin/env python3
# import os
# import torch
# import esm
# from tqdm import tqdm
# import argparse
# import pickle
# from multiprocessing import Pool
# #########esm_env
# # ESM 支持的氨基酸字母表
# VALID_AAS = set("ACDEFGHIKLMNPQRSTVWYBXZO")

# # 不同模型的层数
# MODEL_LAYERS = {
#     "esm2_t6_8M_UR50D": 6,
#     "esm2_t12_35M_UR50D": 12,
#     "esm2_t30_150M_UR50D": 30,
#     "esm2_t33_650M_UR50D": 33,
# }


# def parse_faa(faa_path):
#     """
#     解析并清洗 .faa 文件中的序列：
#       - 合并多行
#       - 去除尾部所有 *, #, +
#       - 将所有非 VALID_AAS 字符替换为 'X'
#     返回列表 [(protein_id, clean_seq), ...]
#     """
#     seqs = []
#     with open(faa_path) as f:
#         header = None
#         seq_lines = []
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             if line.startswith(">"):
#                 # 遇到新 header，先处理前一条
#                 if header is not None and seq_lines:
#                     raw = "".join(seq_lines)
#                     raw = raw.rstrip("*#+")  # 去掉所有尾部 *, #, +
#                     clean = "".join(aa if aa in VALID_AAS else "X" for aa in raw)
#                     seqs.append((header, clean))
#                 # 设置新 header
#                 header = line[1:].split()[0]
#                 seq_lines = []
#             else:
#                 seq_lines.append(line)
#         # 处理最后一条
#         if header is not None and seq_lines:
#             raw = "".join(seq_lines)
#             raw = raw.rstrip("*#+")
#             clean = "".join(aa if aa in VALID_AAS else "X" for aa in raw)
#             seqs.append((header, clean))
#     return seqs


# def process_single_faa(args):
#     faa_file, faa_dir, out_dir, model_name, batch_size, repr_l, device = args

#     if not faa_file.endswith(".faa"):
#         return

#     base = os.path.splitext(faa_file)[0]
#     out_path = os.path.join(out_dir, base + ".pkl")
#     if os.path.exists(out_path):
#         return f"[跳过] {faa_file}"

#     seqs = parse_faa(os.path.join(faa_dir, faa_file))
#     if not seqs:
#         return f"[空文件] {faa_file}"

#     max_layer = MODEL_LAYERS.get(model_name)
#     if max_layer is None:
#         return f"[错误] 不支持模型 {model_name}"
#     if repr_l >= max_layer:
#         repr_l = max_layer - 1

#     # 离线加载 ESM 模型
#     model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
#     model = model.eval().to(device)
#     batch_converter = alphabet.get_batch_converter()

#     embeddings = {}
#     for i in range(0, len(seqs), batch_size):
#         batch = seqs[i : i + batch_size]
#         # batch_converter 只看第二项 sequence
#         try:
#             _, _, tokens = batch_converter(batch)
#         except KeyError as e:
#             return f"[编码失败] {faa_file}: 非法字符 {e}"
#         tokens = tokens.to(device)

#         with torch.no_grad():
#             out = model(tokens, repr_layers=[repr_l], return_contacts=False)
#         reps = out["representations"].get(repr_l)
#         if reps is None:
#             return f"[层错误] {faa_file}: 层 {repr_l} 不存在"

#         for j, (pid, seq) in enumerate(batch):
#             emb = reps[j, 1 : len(seq) + 1].mean(0)
#             embeddings[pid] = emb.cpu()

#     # 保存 pickle
#     with open(out_path, "wb") as f:
#         pickle.dump(embeddings, f)

#     return f"[完成] {faa_file} → {out_path}"


# def main(faa_dir, out_dir, model_name, batch_size, repr_l, device, workers):
#     os.makedirs(out_dir, exist_ok=True)
#     faa_files = [fn for fn in os.listdir(faa_dir) if fn.endswith(".faa")]
#     if repr_l is None:
#         repr_l = MODEL_LAYERS.get(model_name, 30) - 1

#     print(f">> 模型={model_name}, repr_layer={repr_l}, 并行进程={workers}, 文件数={len(faa_files)}")
#     args_list = [
#         (fn, faa_dir, out_dir, model_name, batch_size, repr_l, device)
#         for fn in faa_files
#     ]

#     with Pool(processes=workers) as pool:
#         for res in tqdm(pool.imap_unordered(process_single_faa, args_list), total=len(args_list)):
#             if res:
#                 print(res)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="多进程批量生成 ESM 嵌入 (严格过滤非法字符)")
#     parser.add_argument("--faa-dir", required=True, help="输入 .faa 文件目录")
#     parser.add_argument("--out", required=True, help="输出 .pkl 目录")
#     parser.add_argument(
#         "--model-name",
#         default="esm2_t30_150M_UR50D",
#         help="可选：esm2_t6_8M_UR50D | esm2_t12_35M_UR50D | esm2_t30_150M_UR50D | esm2_t33_650M_UR50D",
#     )
#     parser.add_argument("--batch-size", type=int, default=4, help="每批序列数")
#     parser.add_argument("--repr-l", type=int, default=None, help="表示层索引(默认最后一层)")
#     parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
#     parser.add_argument("--workers", type=int, default=4, help="并行进程数")
#     args = parser.parse_args()

#     main(
#         args.faa_dir,
#         args.out,
#         args.model_name,
#         args.batch_size,
#         args.repr_l,
#         args.device,
#         args.workers,
#     )


#!/usr/bin/env python3
import os
import torch
import esm
from tqdm import tqdm
import argparse
import pickle
from multiprocessing import Pool

######### ESM 环境配置
# 支持的氨基酸字母表
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWYBXZO")
# 模型各层数映射
MODEL_LAYERS = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
}


def parse_faa(faa_path):
    """
    解析并清洗 .faa 文件中的序列：
      - 合并多行
      - 去除尾部所有 *, #, +
      - 将所有非 VALID_AAS 字符替换为 'X'
    返回列表 [(protein_id, clean_seq), ...]
    """
    seqs = []
    with open(faa_path) as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and seq_lines:
                    raw = "".join(seq_lines)
                    raw = raw.rstrip("*#+")
                    clean = "".join(aa if aa in VALID_AAS else "X" for aa in raw)
                    seqs.append((header, clean))
                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None and seq_lines:
            raw = "".join(seq_lines)
            raw = raw.rstrip("*#+")
            clean = "".join(aa if aa in VALID_AAS else "X" for aa in raw)
            seqs.append((header, clean))
    return seqs


def process_single_faa(args):
    faa_file, faa_dir, out_dir, model_name, batch_size, repr_l, device = args

    if not faa_file.endswith(".faa"):
        return None

    base = os.path.splitext(faa_file)[0]
    out_path = os.path.join(out_dir, base + ".pkl")
    if os.path.exists(out_path):
        return f"[跳过] {faa_file} (已存在输出)"

    seqs = parse_faa(os.path.join(faa_dir, faa_file))
    if not seqs:
        return f"[空文件] {faa_file}"

    max_layer = MODEL_LAYERS.get(model_name)
    if max_layer is None:
        return f"[错误] 不支持模型 {model_name}"
    if repr_l is None or repr_l >= max_layer:
        repr_l = max_layer - 1

    # 离线加载模型
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings = {}
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        try:
            _, _, tokens = batch_converter(batch)
        except KeyError as e:
            # 非法字符导致无法编码，跳过整个文件
            return f"[编码失败] {faa_file}: 非法字符 {e}"

        tokens = tokens.to(device)
        try:
            with torch.no_grad():
                out = model(tokens, repr_layers=[repr_l], return_contacts=False)
            reps = out["representations"].get(repr_l)
            if reps is None:
                return f"[层错误] {faa_file}: 层 {repr_l} 不存在"

            for j, (pid, seq) in enumerate(batch):
                emb = reps[j, 1 : len(seq) + 1].mean(0)
                embeddings[pid] = emb.cpu()

        except RuntimeError as e:
            # 捕获 OOM 并跳过本 batch
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"[跳过: 显存溢出] {faa_file} 批次 {i}-{i+len(batch)-1}")
                continue
            else:
                # 其它 RuntimeError，重新抛出
                raise

    # 保存 pickle
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)

    return f"[完成] {faa_file} → {out_path}"


def main(faa_dir, out_dir, model_name, batch_size, repr_l, device, workers):
    os.makedirs(out_dir, exist_ok=True)
    faa_files = [fn for fn in os.listdir(faa_dir) if fn.endswith(".faa")]
    print(f">> 模型={model_name}, repr_layer={repr_l}, 并行进程={workers}, 文件数={len(faa_files)}")
    args_list = [
        (fn, faa_dir, out_dir, model_name, batch_size, repr_l, device)
        for fn in faa_files
    ]

    with Pool(processes=workers) as pool:
        for res in tqdm(pool.imap_unordered(process_single_faa, args_list), total=len(args_list)):
            if res:
                print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多进程批量生成 ESM 嵌入 (支持显存溢出自动跳过)")
    parser.add_argument("--faa-dir", required=True, help="输入 .faa 文件目录")
    parser.add_argument("--out", required=True, help="输出 .pkl 目录")
    parser.add_argument(
        "--model-name",
        default="esm2_t30_150M_UR50D",
        help="可选：esm2_t6_8M_UR50D | esm2_t12_35M_UR50D | esm2_t30_150M_UR50D | esm2_t33_650M_UR50D",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="每批序列数")
    parser.add_argument("--repr-l", type=int, default=None, help="表示层索引(默认最后一层)")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    args = parser.parse_args()

    main(
        args.faa_dir,
        args.out,
        args.model_name,
        args.batch_size,
        args.repr_l,
        args.device,
        args.workers,
    )
