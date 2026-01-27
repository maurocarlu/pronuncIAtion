# 🏗️ L-MAC Architecture Diagram

Questo diagramma illustra l'architettura L-MAC (Listenable Maps for Audio Classifiers) seguendo il concetto di "Progressive Multi-Stage Pipeline", simile alla Figura 4 di VideoChat2.

## 📊 Progressive Stages

```mermaid
flowchart LR
    %% --- STYLES ---
    classDef frozen fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef trainable fill:#FFF3E0,stroke:#EF6C00,stroke-width:3px,color:#E65100;
    classDef data fill:#FFFFFF,stroke:#9E9E9E,stroke-width:1px,rx:5,ry:5;
    classDef operation fill:#F5F5F5,stroke:#616161,stroke-width:1px,stroke-dasharray: 5 5,rx:10,ry:10;
    classDef title fill:none,stroke:none,color:#B71C1C,font-size:16px,font-weight:bold;

    %% ==========================================
    %% STAGE 1: UNIVERSAL INTERFACE
    %% ==========================================
    subgraph S1 ["Stage 1: Universal Backbone Interface"]
        direction LR
        S1_Title[Stage 1]:::title
        
        %% Nodes
        Audio1[("🔊 Raw Audio")]:::data
        Backbone1["❄️ Universal Backbone\n(HuBERT / WavLM / Fusion)\n[FROZEN]"]:::frozen
        Standardizer1["⚙️ Feature\nStandardizer"]:::operation
        Feat1[("Features H")]:::data
        
        %% Connections
        Audio1 --> Backbone1
        Backbone1 --> Standardizer1
        Standardizer1 --> Feat1
    end

    %% ==========================================
    %% STAGE 2: L-MAC TRAINING
    %% ==========================================
    subgraph S2 ["Stage 2: L-MAC Training"]
        direction TB
        S2_Title[Stage 2]:::title
        
        %% Top Row Nodes
        Audio2[("🔊 Audio")]:::data
        Backbone2["❄️ Backbone\n[FROZEN]"]:::frozen
        Feat2[("Features H")]:::data
        
        %% Training Core
        TargetPh["Target\nPhoneme k"]:::data
        LMAC2["🔥 L-MAC Decoder\n(1D U-Net)\n[TRAINABLE]"]:::trainable
        Mask2[("Mask M")]:::data
        
        %% Loss
        LossOp(("Loss L")):::operation
        
        %% Connections
        Audio2 --> Backbone2
        Backbone2 --> Feat2
        Feat2 --> LMAC2
        TargetPh --> LMAC2
        LMAC2 --> Mask2
        
        %% Loss Connections
        Mask2 -.-> LossOp
        Feat2 -.-> LossOp
        
        %% Enforce layout
        Backbone2 ~~~ LMAC2
    end

    %% ==========================================
    %% STAGE 3: INFERENCE & EXPLANATION
    %% ==========================================
    subgraph S3 ["Stage 3: Listenable Map Generation"]
        direction TB
        S3_Title[Stage 3]:::title
        
        %% Input
        Audio3[("🔊 Raw Audio")]:::data
        
        %% Path 1: Mask Gen
        Backbone3["❄️ Backbone"]:::frozen
        LMAC3["❄️ L-MAC Decoder\n[FROZEN]"]:::frozen
        Mask3[("Mask M")]:::data
        
        %% Path 2: Application
        DotProd(("⊙")):::operation
        ListenableMap[("🎧 Listenable\nMap")]:::data
        Human["👤 Human Listener\n/ ASR Check"]:::data
        
        %% Connections
        Audio3 --> Backbone3
        Backbone3 --> LMAC3 --> Mask3
        
        Audio3 --> DotProd
        Mask3 --> DotProd
        DotProd --> ListenableMap
        ListenableMap --> Human
    end
    
    %% Link the subgraphs logically (invisible links for layout)
    S1 ~~~ S2 ~~~ S3
```

### 🗝️ Legenda

- **❄️ Blue Box (Frozen)**: Componenti pre-addestrati i cui pesi non cambiano (Backbone, e Decoder in fase di inferenza). Garantisce che L-MAC sia "plug-and-play".
- **🔥 Orange Box (Trainable)**: L'unico componente che apprende durante la Stage 2 (il Decoder 1D U-Net).
- **Cerchi/Ovali**: Dati in input/output (Audio, Feature, Maschere).
- **Tratteggiato**: Operazioni non parametriche (Loss, Standardizzazione, Moltiplicazione).
