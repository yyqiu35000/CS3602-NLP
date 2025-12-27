import torch

def visualize_mask_logic():
    print("=== Mask Construction Logic Visualization ===")
    
    # 假设场景：
    # Sink = 2 (A, B)
    # Window = 2 (C, D)
    # Current Chunk = 3 (E, F, G)
    # Total Physical Key Length (k_len) = 2 + 2 + 3 = 7
    # Query Length (q_len) = 3 (E, F, G)
    
    q_len = 3
    n_sink = 2
    n_window = 2
    k_len = n_sink + n_window + q_len # 7
    
    print(f"Scenario:")
    print(f"  Query Tokens:   [E, F, G] (q_len={q_len})")
    print(f"  Key Tokens:     [A, B, C, D, E, F, G] (k_len={k_len})")
    print(f"                  |Sink|Wind.| Current |")
    
    # 1. 初始化全 0 Mask (表示全部可见)
    # Shape: [1, 1, q_len, k_len] -> [1, 1, 3, 7]
    mask = torch.zeros((1, 1, q_len, k_len))
    print("\nStep 1: Initialize zeros (Everything Visible)")
    print(mask[0, 0])
    
    # 2. 构建 Causal Mask (上三角)
    # Shape: [3, 3]
    min_val = -100.0 # 用 -100 代表负无穷
    causal_mask = torch.full((q_len, q_len), min_val)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    print("\nStep 2: Create Causal Mask for the Chunk itself")
    print(f"Shape: {causal_mask.shape}")
    print(causal_mask)
    
    # 3. "一步处理": 将 Causal Mask 覆盖到最后 q_len 列
    # new_mask[:, :, :, -q_len:] = causal_mask
    mask[:, :, :, -q_len:] = causal_mask
    
    print(f"\nStep 3: Apply to the last {q_len} columns of the full mask")
    print("Final Mask Structure [3 rows (queries) x 7 cols (keys)]:")
    final_mask = mask[0, 0]
    
    # 打印带标注的矩阵
    print("\n      Keys ->  A    B    C    D    E    F    G")
    print("Queries")
    row_labels = ['E', 'F', 'G']
    for i in range(q_len):
        row_str = f"   {row_labels[i]}          "
        for j in range(k_len):
            val = final_mask[i, j].item()
            if val == 0:
                row_str += " 0.0 " # Visible
            else:
                row_str += "-Inf " # Masked
        print(row_str)

    print("\nAnalysis:")
    print("1. Columns A,B,C,D (Indices 0-3): All 0.0. This means E,F,G can ALL see past tokens (Sink & Window).")
    print("2. Columns E,F,G (Indices 4-6): The causal mask is applied.")
    print("   - Row E can see E (0.0), but cannot see F (-Inf) or G (-Inf).")
    print("   - Row F can see E (0.0) and F (0.0), but cannot see G (-Inf).")
    print("   - Row G can see E, F, and G (All 0.0).")

if __name__ == "__main__":
    visualize_mask_logic()
