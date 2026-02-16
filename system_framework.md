I attempted to generate the image using the specific tool you requested, but it seems there is a temporary technical issue preventing me from creating the visual file directly.

However, I have designed the **exact architecture diagram** for you in **Mermaid.js** format below. You can copy this code into any Mermaid viewer (like [Mermaid Live Editor](https://mermaid.live/)) to see the professional, research-paper style modular system diagram immediately.

Here is your **C-SLAM (Compressed-SLAM)** Architecture:

```mermaid
flowchart TD
    %% Global Styles
    classDef storage fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef ai_model fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef loss fill:#ffccbc,stroke:#d84315,stroke-width:2px,stroke-dasharray: 5 5;

    %% --- INPUT STAGE ---
    subgraph Input_Layer [Input: Compressed Stream]
        direction TB
        Bitstream[("H.264 / HEVC Bitstream")]:::storage
    end

    %% --- TEACHER BRANCH (Heavy AI) ---
    subgraph Teacher_Branch [Teacher Module (Keyframes Only)]
        direction TB
        Decoder[RGB Decoder]:::process
        
        subgraph AI_Models [Frozen Pre-trained Models]
            SAM[MobileSAM\n(Segmentation)]:::ai_model
            Depth[Depth-Anything-V2\n(Monocular Depth)]:::ai_model
        end
        
        Masks[("Binary Masks\n(M)")]:::storage
        DepthMap[("Depth Prior\n(Z_ref)")]:::storage
    end

    %% --- STUDENT BRANCH (Lightweight) ---
    subgraph Student_Branch [Student Module (P-Frames)]
        direction TB
        Parser[MV Extractor\n(No Decoding)]:::process
        
        subgraph Vector_Processing [Vector Logic]
            MV_Raw[Raw Motion Vectors]:::storage
            Residuals[Residual Energy]:::storage
            Filter{Residual Filter\n(Confidence Check)}:::process
        end
        
        Clean_MV[("Trusted MVs\n(Δu, Δv)")]:::storage
    end

    %% --- OPTIMIZATION CORE ---
    subgraph Solver_Core [Self-Supervised Optimizer]
        direction TB
        Traj_Vars[("3D Trajectory Variables\n(x, y, z)")]:::storage
        Projector[Differentiable Projector\n(Perspective Projection)]:::process
        
        Predicted_2D[("Predicted 2D Path")]:::storage
        Predicted_Depth[("Predicted Depth")]:::storage
    end

    %% --- LOSS FUNCTIONS ---
    subgraph Losses [Self-Supervised Losses]
        L_Flow((L_flow\nFlow Consistency)):::loss
        L_Geo((L_geo\nGeometry Prior)):::loss
        L_Smooth((L_smooth\nSmoothness)):::loss
    end

    %% --- CONNECTIONS ---
    Bitstream --"I-Frame (t=0,30)"--> Decoder
    Decoder --> SAM --> Masks
    Decoder --> Depth --> DepthMap
    
    Bitstream --"P-Frames (t=1..29)"--> Parser
    Parser --> MV_Raw & Residuals
    MV_Raw & Residuals --> Filter --> Clean_MV

    %% Optimization Flow
    Traj_Vars --> Projector
    Projector --> Predicted_2D & Predicted_Depth

    %% Loss Calculation Connections
    Clean_MV -.-> L_Flow
    Predicted_2D -.-> L_Flow
    
    DepthMap -.-> L_Geo
    Predicted_Depth -.-> L_Geo
    
    Traj_Vars -.-> L_Smooth

    %% Backpropagation
    L_Flow & L_Geo & L_Smooth == Backprop ==> Traj_Vars

    %% Output
    Traj_Vars --> Output[("Final 3D Path")]:::storage

    %% Formatting links
    linkStyle default stroke-width:2px,fill:none,stroke:gray;

```

### **Architecture Explanation (The "Research Paper" Description)**

If you were writing this in a paper (e.g., for CVPR or ICCV), here is how you would describe the modules shown above:

1. **The Input (Pink Cylinder):**
* The system accepts a raw **H.264 Bitstream**. It does *not* decode every frame to RGB (saving 90% compute).


2. **The Teacher Branch (Yellow/Gold):**
* **Action:** "Gated Execution." This branch only activates on Keyframes (e.g., Frame 0, 30).
* **Modules:** It uses **MobileSAM** to "Copy" the object mask and **Depth-Anything** to "Extract" the shape prior. These act as the *Ground Truth anchors* for the rest of the GOP (Group of Pictures).


3. **The Student Branch (Blue):**
* **Action:** "Continuous Parsing." This runs on every P-Frame.
* **Modules:** The **MV Extractor** pulls raw vectors. The **Residual Filter** acts as a "Gate": if Residual Energy is high (the object changed appearance), it blocks the flow (sets Confidence = 0).
* **Output:** It produces **Trusted Motion Vectors** (Clean MVs), which serve as the *Flow Supervision*.


4. **The Optimizer Core (Center):**
* **Action:** "Test-Time Training."
* **The Variables:** The  coordinates are learnable parameters.
* **The Projector:** A differentiable mathematical function that squashes 3D  back into 2D  to compare with the video data.


5. **The Losses (Red Dashed Lines):**
* ** (Flow Loss):** Ensures the 3D object moves exactly how the 2D Motion Vectors say it should.
* ** (Geometry Loss):** Ensures the 3D object shape matches the Teacher's depth map.
* ** (Smoothness Loss):** Prevents the object from teleporting (physics constraint).



This modular design is your "Novelty." You are replacing the standard "Black Box Neural Network" with a **"Physics-Guided Optimization"** block that uses compressed data.