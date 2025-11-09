# AWS GPU Instance Architecture

This diagram shows the relationships between Terraform modules and AWS resources.

## Module Dependencies

```mermaid
graph TB
    subgraph "Root Module"
        Root[main.tf]
        RootVars[variables.tf]
        RootOutputs[outputs.tf]
    end

    subgraph "Security Group Module"
        SGModule[modules/security_group]
        SGMain[main.tf]
        SGVars[variables.tf]
        SGOutputs[outputs.tf]
    end

    subgraph "EC2 Instance Module"
        EC2Module[modules/ec2_instance]
        EC2Main[main.tf]
        EC2Vars[variables.tf]
        EC2Outputs[outputs.tf]
    end

    Root --> SGModule
    Root --> EC2Module
    SGModule --> SGMain
    SGModule --> SGVars
    SGModule --> SGOutputs
    EC2Module --> EC2Main
    EC2Module --> EC2Vars
    EC2Module --> EC2Outputs

    style Root fill:#e1f5ff
    style SGModule fill:#fff4e1
    style EC2Module fill:#e7f5e1
```

## AWS Resource Architecture

```mermaid
graph LR
    subgraph "AWS VPC: vpc-92dd37ef"
        SG[Security Group<br/>launch-wizard-8]
        
        subgraph "Ingress Rules"
            SSH[SSH: 22<br/>0.0.0.0/0]
            HTTPS[HTTPS: 443<br/>0.0.0.0/0]
            HTTP[HTTP: 80<br/>0.0.0.0/0]
        end
        
        subgraph "EC2 Instance"
            Instance[g4dn.4xlarge<br/>inference-test-server]
            
            subgraph "Instance Details"
                AMI[AMI: ami-0157af9aea2eef346<br/>Amazon Linux 2023]
                GPU[NVIDIA T4 GPU<br/>16GB Memory]
                Key[Key: nwtbootcampkey-pem]
            end
            
            subgraph "Storage"
                Volume[Root Volume<br/>200GB gp3<br/>3000 IOPS<br/>125 MB/s throughput]
            end
            
            subgraph "Network"
                PublicIP[Public IP<br/>Auto-assigned]
                DNS[DNS Hostname<br/>ip-name]
            end
            
            subgraph "Security"
                IMDSv2[IMDSv2 Required<br/>Hop Limit: 2]
            end
        end
    end

    SSH --> SG
    HTTPS --> SG
    HTTP --> SG
    SG --> Instance
    AMI --> Instance
    GPU --> Instance
    Key --> Instance
    Volume --> Instance
    PublicIP --> Instance
    DNS --> Instance
    IMDSv2 --> Instance

    Internet[Internet] --> SSH
    Internet --> HTTPS
    Internet --> HTTP
    
    style SG fill:#ff9999
    style Instance fill:#99ccff
    style Volume fill:#99ff99
    style IMDSv2 fill:#ffcc99
```

## Terraform Module Flow

```mermaid
sequenceDiagram
    participant User
    participant Root as Root Module
    participant SG as Security Group Module
    participant EC2 as EC2 Instance Module
    participant AWS

    User->>Root: terraform apply
    Root->>SG: Create Security Group
    SG->>AWS: Create aws_security_group
    AWS-->>SG: Security Group ID
    SG->>AWS: Create ingress rules
    AWS-->>SG: Rules Created
    SG-->>Root: Return SG ID
    
    Root->>EC2: Create Instance (with SG ID)
    EC2->>AWS: Create aws_instance
    AWS-->>EC2: Instance Created
    EC2-->>Root: Return Instance Details
    
    Root-->>User: Output Instance Info
```

## Data Flow

```mermaid
flowchart TD
    A[terraform.tfvars] --> B[Root Module Variables]
    B --> C{Module Selection}
    C -->|SG Config| D[Security Group Module]
    C -->|EC2 Config| E[EC2 Instance Module]
    
    D --> F[AWS Security Group]
    F --> G[Security Group ID]
    G --> E
    
    E --> H[AWS EC2 Instance]
    H --> I[Instance Outputs]
    
    I --> J[Public IP]
    I --> K[Instance ID]
    I --> L[DNS Name]
    
    style A fill:#e1f5ff
    style F fill:#ff9999
    style H fill:#99ccff
    style J fill:#99ff99
    style K fill:#99ff99
    style L fill:#99ff99
```

## GPU Setup Flow (User Data)

```mermaid
sequenceDiagram
    participant TF as Terraform
    participant EC2 as EC2 Instance
    participant UD as User Data Script
    participant YUM as Package Manager
    participant Docker as Docker Daemon
    participant NVIDIA as NVIDIA Drivers

    TF->>EC2: Launch Instance with User Data
    EC2->>UD: Execute on First Boot
    
    Note over UD: Phase 1: Docker Setup
    UD->>YUM: Install Docker
    YUM-->>UD: Docker Installed
    UD->>Docker: Start & Enable Service
    Docker-->>UD: Service Running
    
    Note over UD: Phase 2: NVIDIA Drivers
    UD->>YUM: Add NVIDIA Repository
    UD->>YUM: Install DKMS + Kernel Headers
    UD->>YUM: Install NVIDIA Drivers
    YUM-->>UD: Drivers Installed
    
    Note over UD: Phase 3: CUDA Toolkit
    UD->>YUM: Install CUDA Toolkit
    YUM-->>UD: CUDA Installed
    UD->>EC2: Configure Environment Variables
    
    Note over UD: Phase 4: Container Toolkit
    UD->>YUM: Install NVIDIA Container Toolkit
    YUM-->>UD: Toolkit Installed
    UD->>Docker: Configure GPU Runtime
    Docker-->>UD: Runtime Configured
    UD->>Docker: Restart Service
    
    Note over UD: Phase 5: Helper Scripts
    UD->>EC2: Create Verification Scripts
    UD->>EC2: Create GPU README
    UD->>EC2: Mark Reboot Required
    
    UD-->>EC2: Setup Complete
    EC2-->>TF: Instance Ready
    
    Note over EC2: Manual Reboot Required
    EC2->>NVIDIA: Load Kernel Modules
    NVIDIA-->>EC2: GPU Available
```

## GPU Setup Components

```mermaid
graph TB
    subgraph "User Data Script"
        A[Update System]
        B[Install Docker]
        C[Install NVIDIA Drivers]
        D[Install CUDA Toolkit]
        E[Install Container Toolkit]
        F[Configure Docker]
        G[Create Helper Scripts]
    end
    
    subgraph "Installed Components"
        H[Docker Engine]
        I[NVIDIA Driver 550+]
        J[CUDA 12.4+]
        K[nvidia-container-toolkit]
    end
    
    subgraph "Helper Scripts"
        L[verify-gpu-setup.sh]
        M[run-gpu-container.sh]
        N[GPU-SETUP-README.txt]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    
    B --> H
    C --> I
    D --> J
    E --> K
    
    G --> L
    G --> M
    G --> N
    
    style H fill:#99ccff
    style I fill:#99ff99
    style J fill:#ffcc99
    style K fill:#ff9999
```
