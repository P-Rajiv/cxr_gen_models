
A slightly more formal version for thesis docs:



```mermaid
flowchart TB
    A[Step 1: Define Structured Clinical Prompts]
    A1[Pathology]
    A2[Severity]
    A3[Location / Laterality]

    B[Step 2: Generate Synthetic Chest X-rays<br/>from Text-conditioned Diffusion Models]

    C[Step 3: Produce Image Findings Text<br/>using MAIRA-2]

    D[Step 4: Convert Prompt and Findings<br/>into Clinical Entity Graphs using RadGraph]

    E[Step 5: Compute Semantic Alignment Score]
    E1[Disease Presence]
    E2[Severity Match]
    E3[Location Match]
    E4[Contradiction Penalty]

    F{Step 6: Selection Gate}
    G[Keep Clinically Consistent Images]
    H[Remove Incorrect / Ambiguous / Noisy Images]

    I[Step 7: Optional Visual Filtering]
    I1[RAD-DINO Similarity]
    I2[Outlier Removal]
    I3[Duplicate Removal]

    J[Step 8: Construct Training Sets]
    J1[Real Only]
    J2[Real + Unfiltered Synthetic]
    J3[Real + Filtered Synthetic]

    K[Step 9: Train Classification Models]

    L[Step 10: Evaluate on Held-out Real Data]
    L1[AUROC]
    L2[F1-score]
    L3[Per-class Performance]

    M[Step 11: Analyze Whether Filtering Improves Downstream Utility]

    A --> A1
    A --> A2
    A --> A3
    A --> B
    B --> C
    C --> D
    A --> D
    D --> E
    E --> E1
    E --> E2
    E --> E3
    E --> E4
    E --> F
    F -- Keep --> G
    F -- Remove --> H
    G --> I
    I --> I1
    I --> I2
    I --> I3
    I --> J
    J --> J1
    J --> J2
    J --> J3
    J --> K
    K --> L
    L --> L1
    L --> L2
    L --> L3
    L --> M
```





### Prompt Generation Strategy

- **Prompting**: Structure clinical prompts by disease category
- **Meta Data**: Use `meta_data.csv` to organize disease labels
- **Sampling**: Collect and sample prompts disease-wise from labeled data

### generated images 
- **diffusion model** : 
- **gans** : 
- **rectified flow**:
 

### report generation of generated images 
- **maira-2**: [maira-2](https://huggingface.co/microsoft/maira-2)
- **code file**: mira_inference.py

### 
