# Modern Data Science Algorithms (2024-2026)

This document covers cutting-edge algorithms and architectures from 2024-2026. These represent the latest advances in machine learning and AI.

---

## Table of Contents

1. [Large Language Models (LLMs)](#large-language-models-llms)
2. [Vision Transformers & Multimodal Models](#vision-transformers--multimodal-models)
3. [Diffusion Models](#diffusion-models)
4. [Efficient Training & Inference](#efficient-training--inference)
5. [Graph Neural Networks (GNN)](#graph-neural-networks-gnn)
6. [Reinforcement Learning](#reinforcement-learning)
7. [Time Series Foundation Models](#time-series-foundation-models)
8. [AutoML & Neural Architecture Search](#automl--neural-architecture-search)

---

## Large Language Models (LLMs)

### 1. GPT-4 / GPT-4o (OpenAI, 2024)
**Architecture**: Transformer-based, multimodal (text + images + audio)

**When to use**:

- Complex reasoning tasks
- Code generation
- Multimodal understanding
- Few-shot learning

**Key Features**:

- 128k context window (GPT-4 Turbo)
- Multimodal input/output
- Function calling capabilities
- Structured output (JSON mode)

**Python Example**:

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a data scientist."},
        {"role": "user", "content": "Explain transformer architecture"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

**Use Cases**:
- Text generation and summarization
- Code completion and debugging
- Data analysis and interpretation
- Few-shot classification

---

### 2. Claude 3 / Claude 3.5 (Anthropic, 2024-2025)
**Architecture**: Constitutional AI + Transformer

**When to use**:
- Long-context tasks (200k+ tokens)
- Complex reasoning
- Code generation
- Analysis of long documents

**Key Features**:
- 200k context window (Claude 3 Opus)
- Extended thinking mode
- Strong coding abilities
- Better refusal of harmful requests

**Python Example**:
```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Analyze this dataset: [long CSV content]"}
    ]
)

print(message.content)
```

**Advantages over GPT-4**:
- Longer context window
- Often better at analysis and reasoning
- More conservative with uncertain information

---

### 3. Llama 3 / 3.1 / 3.2 (Meta, 2024-2025)
**Architecture**: Open-source Transformer (8B, 70B, 405B parameters)

**When to use**:
- Need open-source alternative
- On-premise deployment
- Fine-tuning required
- Cost-sensitive applications

**Key Features**:
- Fully open-source
- Multiple sizes (8B to 405B)
- Can be fine-tuned
- Competitive with closed-source models

**Python Example (via Hugging Face)**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain gradient descent:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Fine-tuning Example (LoRA)**:
```python
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# LoRA configuration for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=16,                    # Rank of update matrices
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Fine-tune with custom data
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        output_dir="./llama3-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True
    )
)
trainer.train()
```

---

### 4. Mistral 7B / Mixtral 8x7B (Mistral AI, 2024)
**Architecture**: Mixture-of-Experts (MoE) Transformer

**When to use**:
- Need efficient inference
- Resource-constrained deployment
- Open-source alternative

**Key Features**:
- Sliding window attention (efficient)
- Mixtral: Sparse MoE (only 2 of 8 experts active per token)
- Apache 2.0 license (fully open)
- Competitive performance with larger models

**Python Example**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    device_map="auto",
    load_in_4bit=True  # Quantization for efficiency
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

messages = [{"role": "user", "content": "Explain transformers"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

**Mixtral Advantage**: 8x7B model performs like 47B but uses only 13B parameters during inference (sparse activation)

---

### 5. Gemini 1.5 / 2.0 (Google, 2024-2025)
**Architecture**: Multimodal Transformer with extremely long context

**Key Features**:
- 1 million token context window (Gemini 1.5 Pro)
- Native multimodal (text, images, video, audio, code)
- Real-time API with streaming
- Function calling and code execution

**When to use**:
- Extremely long documents (entire codebases, books)
- Video understanding
- Multimodal tasks

**Python Example**:
```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel('gemini-1.5-pro')

# Text + Image input
import PIL.Image
img = PIL.Image.open('chart.png')

response = model.generate_content([
    "Analyze this chart and extract key insights:",
    img
])
print(response.text)

# Long context example
with open('entire_codebase.txt', 'r') as f:
    codebase = f.read()  # Can be up to 1M tokens!
    
response = model.generate_content(
    f"Analyze this codebase and suggest improvements:\n\n{codebase}"
)
```

---

## Vision Transformers & Multimodal Models

### 6. Vision Transformer (ViT) v2 (2024)
**Architecture**: Pure transformer for images (no CNN)

**When to use**:
- Image classification with large datasets
- Transfer learning for vision tasks
- Multimodal models (CLIP-style)

**Key Innovation**: Treats images as sequences of patches

**Python Example**:
```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')

image = Image.open('cat.jpg')
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
```

---

### 7. CLIP / CLIP v2 (OpenAI, 2024 updates)
**Architecture**: Contrastive Language-Image Pre-training

**When to use**:
- Zero-shot image classification
- Image search by text
- Image-text matching
- Multimodal embeddings

**Python Example**:
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

image = Image.open("dog.jpg")
text_options = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]

inputs = processor(
    text=text_options, 
    images=image, 
    return_tensors="pt", 
    padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # Image-text similarity scores
probs = logits_per_image.softmax(dim=1)  # Probabilities

print(f"Label probabilities: {probs}")  # [[0.9, 0.05, 0.05]]
```

**Use Cases**:
- Image search: "Find images of sunset over mountains"
- Zero-shot classification: Classify without training
- Content moderation: Detect unsafe images

---

### 8. Segment Anything Model (SAM) (Meta, 2024)
**Architecture**: Vision Transformer + Prompt-based segmentation

**When to use**:
- Image segmentation
- Object detection
- Interactive labeling
- Medical imaging

**Key Feature**: Segment any object with minimal prompting (click, box, text)

**Python Example**:
```python
from segment_anything import sam_model_registry, SamPredictor
import cv2

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

image = cv2.imread('image.jpg')
predictor.set_image(image)

# Segment by point (x, y)
input_point = np.array([[500, 375]])
input_label = np.array([1])  # 1 = foreground

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# masks[0] is the best segmentation mask
```

**Applications**:
- Automated image annotation
- Medical image segmentation
- Video object tracking

---

### 9. DINOv2 (Meta, 2024)
**Architecture**: Self-supervised Vision Transformer

**When to use**:
- Feature extraction from images
- Downstream vision tasks
- Need strong visual representations

**Key Feature**: Trained without labels (self-supervised), excellent features

**Python Example**:
```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

image = Image.open('image.jpg')
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Extract features
features = outputs.last_hidden_state.mean(dim=1)  # [1, 1024]

# Use features for downstream tasks (classification, retrieval, etc.)
```

---

## Diffusion Models

### 10. Stable Diffusion 3 / SDXL (Stability AI, 2024)
**Architecture**: Latent Diffusion Model + Transformer

**When to use**:
- Text-to-image generation
- Image editing
- Inpainting and outpainting

**Key Features**:
- Higher resolution (1024x1024 native)
- Better prompt following
- Faster inference with optimizations

**Python Example**:
```python
from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "A serene landscape with mountains at sunset, oil painting style"
negative_prompt = "blurry, low quality, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=1024,
    width=1024
).images[0]

image.save("generated_landscape.png")
```

**Fine-tuning (LoRA for custom styles)**:
```python
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig

# Load base model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3")

# Apply LoRA for custom style (e.g., "your art style")
pipe.load_lora_weights("path/to/your_style_lora")

# Generate with custom style
image = pipe("Portrait of a woman in my custom style").images[0]
```

---

### 11. DALL-E 3 (OpenAI, 2024)
**Architecture**: Diffusion Model with enhanced text understanding

**When to use**:
- High-quality image generation
- Complex prompt following
- Commercial applications

**Key Features**:
- Better text rendering in images
- Improved prompt adherence
- Safer content generation

**Python Example**:
```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.images.generate(
    model="dall-e-3",
    prompt="A futuristic city with flying cars and neon lights",
    size="1024x1024",
    quality="hd",
    n=1
)

image_url = response.data[0].url
```

---

### 12. Consistency Models (2024)
**Architecture**: Fast diffusion sampling (1-step generation)

**When to use**:
- Need fast inference
- Real-time image generation
- Resource-constrained environments

**Key Innovation**: Generate images in 1-4 steps vs 50-100 for standard diffusion

**Python Example**:
```python
from diffusers import ConsistencyModelPipeline
import torch

pipe = ConsistencyModelPipeline.from_pretrained(
    "openai/consistency-model-sd",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate in 1 step (vs 50+ for normal diffusion)
image = pipe(
    prompt="A beautiful sunset",
    num_inference_steps=1  # ⚡ Super fast!
).images[0]
```

**Speed Comparison**:
- Standard diffusion: 50 steps, ~5 seconds
- Consistency model: 1 step, ~0.2 seconds (25x faster!)

---

## Efficient Training & Inference

### 13. LoRA (Low-Rank Adaptation, 2024 improvements)
**Architecture**: Parameter-efficient fine-tuning

**When to use**:
- Fine-tune large models with limited GPU memory
- Multiple task-specific adapters
- Rapid experimentation

**Key Benefit**: Fine-tune 1% of parameters, achieve 99% of full fine-tuning performance

**Python Example**:
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# LoRA configuration
lora_config = LoraConfig(
    r=8,                      # Rank (lower = fewer parameters)
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Wrap model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 8,030,261,248 || trainable%: 0.05%

# Train as normal
# After training, can merge LoRA weights or keep separate for multi-task
```

**Advantages**:
- 90% less memory for fine-tuning
- Multiple adapters can be swapped (multi-task)
- Faster training

---

### 14. QLoRA (Quantized LoRA, 2024)
**Architecture**: LoRA + 4-bit quantization

**When to use**:
- Fine-tune on single GPU (even 70B models)
- Extremely limited resources

**Key Innovation**: Fine-tune 70B model on 24GB GPU!

**Python Example**:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Fine-tune 70B model on 24GB GPU! 🚀
```

---

### 15. Flash Attention 2 / 3 (2024-2025)
**Architecture**: Optimized attention computation

**When to use**:
- Training or inference with transformers
- Long sequences
- Memory-constrained scenarios

**Key Benefit**: 2-4x faster attention, uses less memory

**Python Example (integrated in Transformers)**:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",  # Enable Flash Attention
    torch_dtype=torch.float16,
    device_map="auto"
)

# Now all attention operations use Flash Attention 2
# 2-4x faster, especially for long sequences!
```

**Impact**:
- 2-4x faster training
- 50% less memory usage
- Enables longer context windows

---

### 16. Grouped Query Attention (GQA, 2024)
**Architecture**: Efficient alternative to Multi-Head Attention

**When to use**:
- Large-scale models
- Long-context applications
- Inference optimization

**Key Innovation**: Share key/value heads across query heads

**Used in**: Llama 3, Mistral, Gemini

**Comparison**:
- Multi-Head Attention (MHA): All heads independent
- Multi-Query Attention (MQA): Share K/V across all queries (fastest, lowest quality)
- Grouped Query Attention (GQA): Share K/V within groups (best trade-off)

---

## Graph Neural Networks (GNN)

### 17. Graph Transformer Networks (2024)
**Architecture**: Transformer applied to graph structures

**When to use**:
- Social network analysis
- Molecular property prediction
- Recommendation systems
- Knowledge graphs

**Python Example (using PyTorch Geometric)**:
```python
import torch
from torch_geometric.nn import TransformerConv
from torch_geometric.datasets import Planetoid

# Load graph dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GraphTransformer(
    in_channels=dataset.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
)

# Train on graph
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

**Applications**:
- Drug discovery (molecule graphs)
- Social network analysis
- Traffic prediction
- Recommendation systems

---

## Reinforcement Learning (Modern, 2024-2026)

### 18. Proximal Policy Optimization (PPO) - 2024 improvements
**Architecture**: On-policy RL algorithm (used in ChatGPT RLHF)

**When to use**:
- Fine-tuning LLMs with human feedback (RLHF)
- Robotics control
- Game playing

**Python Example (using TRL for LLM fine-tuning)**:
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# Load model with value head for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# PPO configuration
ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1e-5,
    ppo_epochs=4,
    mini_batch_size=4
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer
)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        query_tensors = batch["input_ids"]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        
        # Compute rewards (from human feedback or reward model)
        rewards = compute_rewards(query_tensors, response_tensors)
        
        # Update model with PPO
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

**Applications**:
- RLHF for LLMs (ChatGPT, Claude)
- Robotics manipulation
- Game AI

---

### 19. Direct Preference Optimization (DPO, 2024)
**Architecture**: Simpler alternative to RLHF (no reward model needed)

**When to use**:
- Fine-tune LLMs with preference data
- Simpler than full RLHF
- More stable training

**Key Benefit**: Skip reward model training, directly optimize from preferences

**Python Example**:
```python
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# Preference dataset format:
# {"prompt": "...", "chosen": "good response", "rejected": "bad response"}

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will create reference model automatically
    beta=0.1,         # KL penalty coefficient
    train_dataset=preference_dataset,
    tokenizer=tokenizer
)

dpo_trainer.train()
```

**Advantages over PPO**:
- Simpler (no reward model)
- More stable
- Faster training

---

### 20. Soft Actor-Critic (SAC) - 2024 Extensions
**Paper**: Haarnoja et al. (2018), 2024 improvements
**Architecture**: Off-policy, maximum entropy RL

**When to use**:
- Continuous control tasks (robotics)
- Sample-efficient learning
- Need both exploration and exploitation

**Key Innovation**: Maximize reward + entropy (encourages diverse behaviors)

**Python Example**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        # Twin Q-networks (reduce overestimation)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        # Gaussian policy network
        self.policy = GaussianPolicyNetwork(state_dim, action_dim, hidden_dim)
        
        # Automatic temperature tuning
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_entropy = -action_dim  # Heuristic
        
        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=3e-4
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
    
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if deterministic:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Sample next actions
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            
            # Target Q-values
            q1_next_target = self.q1_target(next_state, next_action)
            q2_next_target = self.q2_target(next_state, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # Add entropy term
            next_q_value = reward + (1 - done) * 0.99 * (
                min_q_next_target - self.log_alpha.exp() * next_log_pi
            )
        
        # Q-function loss
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Policy loss
        new_action, log_pi, _ = self.policy.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.log_alpha.exp() * log_pi - min_q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Temperature loss (automatic tuning)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
```

**Production Use**:
- **Boston Dynamics**: Quadruped locomotion
- **Tesla**: Autonomous driving components
- **DeepMind**: Robotic manipulation

---

### 21. Twin Delayed DDPG (TD3) - 2024 Enhancements
**Paper**: Fujimoto et al. (2018), 2024 improvements
**Architecture**: Deterministic policy gradient with stability improvements

**Key Innovations**:
1. Twin Q-networks (reduce overestimation)
2. Delayed policy updates
3. Target policy smoothing

**When to use**:
- Continuous control (alternative to SAC)
- More deterministic policies preferred
- Simpler than SAC (no entropy term)

**Python Example**:
```python
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        # Twin critics
        self.critic_1 = CriticNetwork(state_dim, action_dim)
        self.critic_2 = CriticNetwork(state_dim, action_dim)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        
        # Deterministic actor
        self.actor = ActorNetwork(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.max_action = max_action
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2  # Update policy every 2 critic updates
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=3e-4
        )
        
        self.total_it = 0
    
    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if noise != 0:
            action = action + np.random.normal(0, self.max_action * noise, size=action.shape)
        
        return action.clip(-self.max_action, self.max_action)
    
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        
        # Sample batch
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Target policy smoothing: add noise to target action
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute target Q-value (take minimum of twin Q-networks)
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * 0.99 * target_Q
        
        # Update critics
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy update
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
```

**TD3 vs SAC**:
- TD3: Deterministic, simpler, no entropy term
- SAC: Stochastic, maximum entropy, better exploration

---

### 22. Offline RL / Conservative Q-Learning (CQL) - 2024
**Paper**: Kumar et al. (2020), 2024 improvements
**Architecture**: Learn from fixed datasets without environment interaction

**When to use**:
- No environment access (historical data only)
- Safety-critical domains (healthcare, autonomous driving)
- Expensive/slow environment interactions

**Key Challenge**: Overestimation on out-of-distribution (OOD) actions

**Python Example**:
```python
class CQLAgent:
    def __init__(self, state_dim, action_dim, cql_weight=1.0):
        self.q_network = QNetwork(state_dim, action_dim)
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.cql_weight = cql_weight
        
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=3e-4)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
    
    def cql_loss(self, state, action, reward, next_state, done):
        # Standard Bellman error
        with torch.no_grad():
            next_action, _ = self.policy.sample(next_state)
            target_q = reward + (1 - done) * 0.99 * self.q_network(next_state, next_action)
        
        current_q = self.q_network(state, action)
        bellman_loss = F.mse_loss(current_q, target_q)
        
        # CQL penalty: lower Q-values for OOD actions
        # Sample random actions (OOD)
        random_actions = torch.rand_like(action) * 2 - 1
        random_q = self.q_network(state, random_actions)
        
        # Sample policy actions
        policy_actions, _ = self.policy.sample(state)
        policy_q = self.q_network(state, policy_actions)
        
        # CQL loss: maximize Q on dataset actions, minimize on OOD
        cql_loss = (torch.logsumexp(random_q, dim=0) - current_q.mean() +
                    torch.logsumexp(policy_q, dim=0) - current_q.mean())
        
        total_loss = bellman_loss + self.cql_weight * cql_loss
        return total_loss
    
    def train(self, offline_dataset, num_epochs=100):
        for epoch in range(num_epochs):
            for batch in offline_dataset:
                loss = self.cql_loss(**batch)
                
                self.q_optimizer.zero_grad()
                loss.backward()
                self.q_optimizer.step()
```

**Production Use**:
- **Waymo**: Learning from logged driving data
- **Healthcare**: Clinical decision support from patient records
- **Recommender systems**: Learning from historical user interactions

---

### 23. Decision Transformer (2024 improvements)
**Paper**: Chen et al. (2021), 2024 extensions
**Architecture**: Sequence modeling for RL (treat RL as language modeling)

**Key Innovation**: Frame RL as supervised sequence prediction

**When to use**:
- Offline RL with transformer benefits
- Long-horizon planning
- Multi-task generalization

**Python Example**:
```python
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, n_layer=3, n_head=1):
        super().__init__()
        
        # Token embeddings
        self.embed_timestep = nn.Embedding(1000, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(action_dim, hidden_size)
        
        # Transformer encoder
        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4*hidden_size,
            activation_function='relu'
        )
        self.transformer = GPT2Model(config)
        
        # Prediction heads
        self.predict_action = nn.Linear(hidden_size, action_dim)
        self.predict_return = nn.Linear(hidden_size, 1)
        self.predict_state = nn.Linear(hidden_size, state_dim)
    
    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embed inputs
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        
        # Stack: (return_1, state_1, action_1, return_2, state_2, action_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_len, -1)
        
        # Transformer forward
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        x = transformer_outputs['last_hidden_state']
        
        # Reshape and get predictions
        x = x.reshape(batch_size, seq_len, 3, -1).permute(0, 2, 1, 3)
        
        # Predict actions from states
        action_preds = self.predict_action(x[:, 1])  # State positions
        
        return action_preds
    
    def get_action(self, states, actions, returns_to_go, timesteps):
        # Predict next action conditioned on desired return
        states = torch.FloatTensor(states).reshape(1, -1, states.shape[-1])
        actions = torch.FloatTensor(actions).reshape(1, -1, actions.shape[-1])
        returns_to_go = torch.FloatTensor(returns_to_go).reshape(1, -1, 1)
        timesteps = torch.LongTensor(timesteps).reshape(1, -1)
        
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        
        return action_preds[0, -1].detach().cpu().numpy()

# Training
model = DecisionTransformer(state_dim=17, action_dim=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in offline_dataset:
    states, actions, returns_to_go, timesteps = batch
    
    # Predict actions
    action_preds = model(states, actions, returns_to_go, timesteps)
    
    # Supervised loss
    loss = F.mse_loss(action_preds, actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Advantages**:
- Simple (supervised learning, no value functions)
- Long-horizon credit assignment (via transformer)
- Goal-conditioned (specify desired return)

**Production Use**:
- **Meta**: Multi-task robotics
- **Microsoft**: Game AI

---

### 24. Multi-Agent RL (MARL) - 2024 SOTA
**Popular Algorithms**:
- **MAPPO**: Multi-Agent PPO
- **QMIX**: Value decomposition for cooperation
- **MADDPG**: Multi-Agent DDPG

**When to use**:
- Multiple interacting agents
- Cooperative, competitive, or mixed scenarios
- Swarm robotics, traffic control

**Python Example (MAPPO)**:

```python
class MAPPOAgent:
    def __init__(self, n_agents, obs_dim, action_dim, state_dim):
        self.n_agents = n_agents
        
        # Decentralized actors (each agent has own policy)
        self.actors = [
            ActorNetwork(obs_dim, action_dim) for _ in range(n_agents)
        ]
        
        # Centralized critic (uses global state)
        self.critic = CentralizedCritic(state_dim)
        
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=5e-4) for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
    
    def select_actions(self, observations):
        """Each agent selects action based on local observation"""
        actions = []
        log_probs = []
        
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs)
            action_probs = self.actors[i](obs_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
        
        return actions, log_probs
    
    def update(self, rollout_buffer):
        """Update using centralized training, decentralized execution (CTDE)"""
        states, observations, actions, rewards, dones = rollout_buffer.get()
        
        # Compute returns and advantages using centralized value
        values = self.critic(states)
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Update each actor with PPO
        for agent_id in range(self.n_agents):
            agent_obs = observations[:, agent_id]
            agent_actions = actions[:, agent_id]
            
            # ... PPO update for this agent's policy
            # (similar to single-agent PPO but using centralized critic)
        
        # Update centralized critic
        value_loss = F.mse_loss(self.critic(states), returns)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
```

**Production Use**:
- **OpenAI**: Hide and Seek, Dota 2
- **DeepMind**: StarCraft II (AlphaStar)
- **Traffic systems**: Cooperative signal control

---

### 25. RLHF Alternatives (2024-2026)
**Beyond DPO and PPO**:

#### Constitutional AI (CAI) - Anthropic
- Self-critique and self-improvement
- No human labelers needed (after initial principles)

#### RLAIF (RL from AI Feedback)
- Replace human feedback with AI feedback
- More scalable

#### Kahneman-Tversky Optimization (KTO)
- Based on prospect theory
- More aligned with human preferences

**Python Example (RLAIF)**:
```python
# Use strong model to provide feedback for weak model
strong_model = AutoModelForCausalLM.from_pretrained("gpt-4")
weak_model = AutoModelForCausalLM.from_pretrained("llama-7b")

def ai_feedback(prompt, response):
    """Strong model scores weak model's response"""
    eval_prompt = f"Rate this response (0-10):\nPrompt: {prompt}\nResponse: {response}\nScore:"
    score = strong_model.generate(eval_prompt)
    return float(score)

# Train weak model with AI feedback (instead of human feedback)
for batch in dataset:
    prompts = batch['prompts']
    responses = weak_model.generate(prompts)
    rewards = [ai_feedback(p, r) for p, r in zip(prompts, responses)]
    
    # Standard PPO update
    ppo_trainer.step(prompts, responses, rewards)
```

---

## Time Series Foundation Models

### 20. TimeGPT / Lag-Llama (2024)
**Architecture**: Transformer-based foundation models for time series

**When to use**:
- Time series forecasting
- Zero-shot prediction on new series
- Multiple related time series

**Key Innovation**: Pre-trained on thousands of time series, few-shot adaptation

**Python Example (TimeGPT via Nixtla)**:
```python
from nixtla import TimeGPT

timegpt = TimeGPT(token="your-token")

# Zero-shot forecasting
forecast = timegpt.forecast(
    df=historical_data,      # Your time series
    h=12,                    # Forecast 12 steps ahead
    freq='M',                # Monthly frequency
    finetune_steps=0         # Zero-shot!
)

# Or fine-tune on your data
forecast_finetuned = timegpt.forecast(
    df=historical_data,
    h=12,
    finetune_steps=50       # Fine-tune for 50 steps
)
```

**Advantages**:
- No feature engineering
- Works out-of-box on new series
- Beats ARIMA, Prophet on many benchmarks

---

### 21. Temporal Fusion Transformer (TFT) - 2024 updates
**Architecture**: Attention-based time series forecasting

**When to use**:
- Multi-horizon forecasting
- Multiple covariates
- Interpretability needed

**Python Example (using PyTorch Forecasting)**:
```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Prepare dataset
training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="sales",
    group_ids=["store_id"],
    min_encoder_length=12,
    max_encoder_length=24,
    min_prediction_length=1,
    max_prediction_length=6,
    static_categoricals=["store_type"],
    time_varying_known_reals=["promotion", "holiday"],
    time_varying_unknown_reals=["sales"]
)

# Train TFT
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16
)

trainer = pl.Trainer(max_epochs=30, gpus=1)
trainer.fit(tft, train_dataloaders=train_dataloader)

# Forecast
predictions = tft.predict(test_data)
```

---

## AutoML & Neural Architecture Search

### 22. AutoGluon Tabular (2024)
**Architecture**: Automated ensemble of best algorithms

**When to use**:
- Tabular data
- Need baseline quickly
- Limited ML expertise

**Python Example**:
```python
from autogluon.tabular import TabularPredictor

# One-line training!
predictor = TabularPredictor(label='target', eval_metric='roc_auc')
predictor.fit(train_data, time_limit=3600)  # 1 hour

# Automatically tries: XGBoost, LightGBM, CatBoost, NN, etc.
# Builds ensemble of best models

# Predict
predictions = predictor.predict(test_data)

# Leaderboard
leaderboard = predictor.leaderboard()
print(leaderboard)
```

**What it does automatically**:
- Feature preprocessing
- Model selection
- Hyperparameter tuning
- Ensemble creation

---

### 23. H2O AutoML (2024)
**Architecture**: Automated ML pipeline

**Python Example**:
```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Load data
df = h2o.import_file("data.csv")

# Split
train, test = df.split_frame(ratios=[0.8])

# AutoML
aml = H2OAutoML(max_runtime_secs=3600, seed=42)
aml.train(x=features, y='target', training_frame=train)

# Leaderboard
lb = aml.leaderboard
print(lb.head())

# Best model
best_model = aml.leader
predictions = best_model.predict(test)
```

---

## Emerging Techniques

### 24. Retrieval-Augmented Generation (RAG, 2024 improvements)
**Architecture**: LLM + Vector database for external knowledge

**When to use**:
- Need up-to-date information
- Domain-specific knowledge
- Reduce hallucination

**Python Example (using LangChain + ChromaDB)**:
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Load documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create retrieval QA chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Query with retrieval
result = qa_chain({"query": "What is the company's revenue for 2024?"})
print(result["result"])
print("Sources:", result["source_documents"])
```

**Advanced RAG (2024)**:
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI

# Load documents
documents = SimpleDirectoryReader('data').load_data()

# Create index with chunking and metadata
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    chunk_size=512,
    chunk_overlap=50
)

index = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context
)

# Query engine with hybrid search (vector + keyword)
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"
)

response = query_engine.query("Summarize the key findings")
print(response)
```

---

### 25. Mamba (State Space Model, 2024)
**Architecture**: Alternative to Transformers for sequences

**When to use**:
- Very long sequences (>100k tokens)
- Linear-time complexity needed
- Resource-constrained

**Key Innovation**: O(n) complexity vs O(n²) for transformers

**Python Example**:
```python
from mamba_ssm import Mamba

model = Mamba(
    d_model=256,      # Model dimension
    d_state=16,       # SSM state dimension
    d_conv=4,         # Convolution kernel size
    expand=2          # Expansion factor
)

# Forward pass
x = torch.randn(batch_size, seq_len, d_model)
output = model(x)  # O(n) complexity!
```

**Advantages over Transformers**:
- Linear complexity (vs quadratic)
- Better for very long sequences
- Less memory usage

**Trade-offs**:
- Newer, less tested
- Fewer pre-trained models

---

## Algorithm Selection Guide (2024-2026)

```
Task: Text Generation
  → ChatGPT API? → GPT-4o or Claude 3.5
  → On-premise? → Llama 3 or Mistral
  → Need speed? → Llama 3.2 8B + LoRA

Task: Image Generation
  → High quality? → DALL-E 3
  → Open source? → Stable Diffusion 3
  → Need speed? → Consistency Models

Task: Vision Tasks
  → Image classification? → ViT or DINOv2
  → Segmentation? → SAM
  → Zero-shot? → CLIP

Task: Tabular Data (still classic!)
  → XGBoost / LightGBM / CatBoost
  → Or AutoGluon for automation

Task: Time Series
  → Traditional? → ARIMA, Prophet
  → Modern? → TimeGPT, TFT

Task: Fine-tuning Large Models
  → Limited GPU? → QLoRA
  → Multiple tasks? → LoRA adapters
  → Simplest? → DPO (preference optimization)

Task: Recommendation Systems
  → Large-scale retrieval? → Two-Tower Model
  → Ranking optimization? → DeepFM, DCN
  → Sequential behavior? → GRU4Rec, SASRec
  → Multi-objective? → Multi-Task Learning (MMoE)
```

---

## 24. Two-Tower Model (Dual Encoder for Recommendations)

### When to Use
- **Large-scale candidate retrieval** (millions to billions of items)
- Need fast approximate nearest neighbor search
- Separate user and item representations
- Real-time personalization at scale (YouTube, TikTok, Pinterest)

### Pros
- **Scalable**: Can pre-compute item embeddings offline
- **Fast inference**: ANN search in milliseconds
- **Flexible**: Can add any features to user/item towers
- **Transfer learning**: Pre-trained embeddings useful for other tasks

### Cons
- **Two-stage paradigm**: Retrieval + ranking (not end-to-end)
- **Dot product limitation**: Only captures linear interactions
- **Cold start**: New items need embedding updates
- **Feature interactions**: Cannot model user-item feature crosses

### Architecture

```
User Features              Item Features
    ↓                          ↓
[User Tower]              [Item Tower]
 (DNN/Transformer)         (DNN/Transformer)
    ↓                          ↓
User Embedding (128d)     Item Embedding (128d)
    └────────→ Dot Product ←─────────┘
                  ↓
            Similarity Score
```

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=128, hidden_dims=[256, 128]):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # User tower
        self.user_embedding = nn.Embedding(n_users, 64)
        user_layers = []
        in_dim = 64
        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        user_layers.append(nn.Linear(in_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # Item tower
        self.item_embedding = nn.Embedding(n_items, 64)
        item_layers = []
        in_dim = 64
        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        item_layers.append(nn.Linear(in_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)
        
        # L2 normalize embeddings
        self.temperature = 0.05  # Temperature for contrastive learning
    
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Pass through towers
        user_vec = self.user_tower(user_emb)
        item_vec = self.item_tower(item_emb)
        
        # L2 normalize
        user_vec = F.normalize(user_vec, p=2, dim=1)
        item_vec = F.normalize(item_vec, p=2, dim=1)
        
        return user_vec, item_vec
    
    def get_user_embedding(self, user_ids):
        """Get user embeddings for inference"""
        user_emb = self.user_embedding(user_ids)
        user_vec = self.user_tower(user_emb)
        return F.normalize(user_vec, p=2, dim=1)
    
    def get_item_embedding(self, item_ids):
        """Get item embeddings for inference (can be pre-computed)"""
        item_emb = self.item_embedding(item_ids)
        item_vec = self.item_tower(item_emb)
        return F.normalize(item_vec, p=2, dim=1)

# Contrastive loss (in-batch negatives)
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, user_vec, item_vec):
        """
        user_vec: (batch_size, embedding_dim)
        item_vec: (batch_size, embedding_dim)
        """
        # Compute similarity matrix: (batch_size, batch_size)
        logits = torch.matmul(user_vec, item_vec.T) / self.temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Cross-entropy loss (treats all off-diagonal as negatives)
        loss = F.cross_entropy(logits, labels)
        return loss

# Training example
n_users = 10000
n_items = 5000
batch_size = 512
embedding_dim = 128

model = TwoTowerModel(n_users, n_items, embedding_dim)
criterion = ContrastiveLoss(temperature=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Simulated training data
user_ids = torch.randint(0, n_users, (batch_size,))
item_ids = torch.randint(0, n_items, (batch_size,))

# Training step
model.train()
optimizer.zero_grad()
user_vec, item_vec = model(user_ids, item_ids)
loss = criterion(user_vec, item_vec)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")

# Inference: Find top-k items for a user
model.eval()
with torch.no_grad():
    # Get user embedding
    test_user = torch.tensor([0])
    user_emb = model.get_user_embedding(test_user)  # (1, 128)
    
    # Get all item embeddings (in production, use FAISS for ANN search)
    all_item_ids = torch.arange(100)  # First 100 items for demo
    item_embs = model.get_item_embedding(all_item_ids)  # (100, 128)
    
    # Compute similarities
    scores = torch.matmul(user_emb, item_embs.T).squeeze()  # (100,)
    
    # Top-10 items
    top_k = 10
    top_scores, top_indices = torch.topk(scores, k=top_k)
    
    print(f"\nTop-{top_k} recommendations for user 0:")
    for rank, (item_id, score) in enumerate(zip(top_indices, top_scores), 1):
        print(f"  {rank}. Item {item_id.item()}: score {score.item():.4f}")
```

### Production Deployment

```python
# Using FAISS for billion-scale ANN search
import faiss
import numpy as np

def build_item_index(model, n_items, embedding_dim=128):
    """Pre-compute and index all item embeddings"""
    model.eval()
    
    # Generate all item embeddings
    batch_size = 1000
    all_item_embs = []
    
    with torch.no_grad():
        for start_idx in range(0, n_items, batch_size):
            end_idx = min(start_idx + batch_size, n_items)
            item_ids = torch.arange(start_idx, end_idx)
            item_embs = model.get_item_embedding(item_ids)
            all_item_embs.append(item_embs.cpu().numpy())
    
    item_embeddings = np.vstack(all_item_embs).astype('float32')
    
    # Build FAISS index (inner product search)
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product = cosine for normalized vectors
    index.add(item_embeddings)
    
    return index, item_embeddings

def retrieve_top_k(user_embedding, item_index, k=100):
    """Fast retrieval using FAISS"""
    user_emb_np = user_embedding.cpu().numpy().astype('float32')
    scores, indices = item_index.search(user_emb_np, k)
    return indices[0], scores[0]

# Build index (done offline, once per day)
item_index, item_embs = build_item_index(model, n_items, embedding_dim)

# Online inference
with torch.no_grad():
    user_id = torch.tensor([42])
    user_emb = model.get_user_embedding(user_id)
    top_items, top_scores = retrieve_top_k(user_emb, item_index, k=100)
    
print(f"Retrieved top-100 items in <1ms using FAISS")
```

### When NOT to Use
- Small catalog (< 1000 items) → Use simple collaborative filtering
- Need fine-grained user-item interactions → Use ranking models (DCN, DeepFM)
- Real-time item updates critical → Two-tower requires re-indexing

### Real-World Applications
- **YouTube**: Deep Neural Networks for YouTube Recommendations (2016)
- **TikTok**: For You page candidate generation
- **Pinterest**: Visual search and recommendations
- **Airbnb**: Listing embeddings for search ranking

### Key Insights
- **Candidate generation**: Two-tower retrieves top 100-1000 from millions
- **Follow-up ranking**: Use cross-feature models (DCN) for final ranking
- **Hard negatives**: Sample hard negatives (popular items user didn't click) for better training
- **Temperature scaling**: Lower temperature (0.01-0.05) improves contrastive learning

---

## 25. Deep & Cross Network (DCN)

### When to Use
- **CTR prediction** with sparse categorical features
- Need explicit feature crosses (user × item, city × category)
- Combining memorization (cross) and generalization (deep)
- Web-scale recommendation and ranking

### Pros
- **Automatic feature crosses**: Learns polynomial feature interactions
- **Efficient**: Polynomial complexity, not exponential
- **Complementary**: Cross network + deep network = best of both worlds
- **State-of-art**: Outperforms Wide & Deep, DeepFM on many benchmarks

### Cons
- **Complexity**: More hyperparameters than simple models
- **Training time**: Slower than linear models
- **Overfitting risk**: Needs regularization on sparse features

### Architecture

```
    Input Features (Sparse + Dense)
           ↓
    [Embedding Layer]
           ↓
    ┌──────────────────┐
    │  Cross Network   │  Deep Network
    │   (Explicit      │  (Implicit
    │   Polynomial     │   Nonlinear
    │   Crosses)       │   Transforms)
    └──────────────────┘
           ↓
    [Concatenate]
           ↓
    [Output Layer]
           ↓
        CTR Score
```

### Python Implementation

```python
import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Weight and bias for each cross layer
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])
    
    def forward(self, x0):
        """
        x0: (batch_size, input_dim)
        Cross layer: x_{l+1} = x0 * (w_l^T * x_l) + b_l + x_l
        """
        x_l = x0
        for i in range(self.num_layers):
            w = self.weights[i]
            b = self.biases[i]
            
            # x0 * (w^T * x_l): outer product then weight
            x_l = x0 * torch.matmul(x_l, w) + b + x_l
        
        return x_l

class DeepCrossNetwork(nn.Module):
    def __init__(self, num_features, embedding_dim=16, 
                 deep_layers=[256, 128, 64], cross_layers=3):
        super().__init__()
        
        # Assume features are categorical (need embedding)
        self.embedding = nn.Embedding(num_features, embedding_dim)
        input_dim = num_features * embedding_dim
        
        # Cross network
        self.cross_net = CrossNetwork(input_dim, cross_layers)
        
        # Deep network
        deep_net_layers = []
        in_dim = input_dim
        for hidden_dim in deep_layers:
            deep_net_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        self.deep_net = nn.Sequential(*deep_net_layers)
        
        # Combination layer
        self.output_layer = nn.Linear(input_dim + deep_layers[-1], 1)
    
    def forward(self, x):
        """
        x: (batch_size, num_features) - categorical feature indices
        """
        # Embed features
        emb = self.embedding(x)  # (batch_size, num_features, embedding_dim)
        emb_flat = emb.view(emb.size(0), -1)  # (batch_size, num_features * embedding_dim)
        
        # Cross network
        cross_out = self.cross_net(emb_flat)
        
        # Deep network
        deep_out = self.deep_net(emb_flat)
        
        # Concatenate and predict
        combined = torch.cat([cross_out, deep_out], dim=1)
        logit = self.output_layer(combined)
        
        return torch.sigmoid(logit.squeeze())

# Training example
num_features = 50  # Number of categorical features
batch_size = 512

model = DeepCrossNetwork(
    num_features=num_features,
    embedding_dim=16,
    deep_layers=[256, 128, 64],
    cross_layers=3
)

# Simulated data (sparse categorical features)
x = torch.randint(0, num_features, (batch_size, num_features))
y = torch.randint(0, 2, (batch_size,)).float()  # Binary labels (click/no-click)

# Training
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
optimizer.zero_grad()
predictions = model(x)
loss = criterion(predictions, y)
loss.backward()
optimizer.step()

print(f"DCN Training loss: {loss.item():.4f}")

# Inference
model.eval()
with torch.no_grad():
    test_x = torch.randint(0, num_features, (10, num_features))
    ctr_predictions = model(test_x)
    print(f"\nCTR predictions (first 10): {ctr_predictions.numpy()}")
```

### When NOT to Use
- Small datasets (< 10K samples) → Too complex, use logistic regression
- Need interpretability → Use linear models or decision trees
- Extremely high-dimensional sparse features → Consider feature selection first

### Real-World Applications
- **Google**: CTR prediction for ads
- **Criteo**: Display advertising CTR
- **Facebook**: News feed ranking
- **E-commerce**: Product ranking on search/browse pages

---

## 26. Neural Collaborative Filtering (NCF)

### When to Use
- User-item interaction data (implicit feedback: clicks, views, purchases)
- Need to capture **nonlinear** user-item interactions
- Combining matrix factorization with neural networks
- Medium to large-scale recommendation (thousands to millions of users/items)

### Pros
- **Nonlinear modeling**: Captures complex interaction patterns
- **Flexible**: Can incorporate user/item features
- **Better than MF**: Outperforms traditional matrix factorization
- **Interpretable**: Combines GMF (generalized MF) and MLP

### Cons
- **Data hungry**: Needs more data than simple CF
- **Training time**: Slower than matrix factorization
- **Cold start**: Still struggles with new users/items without features

### Architecture

```
User ID          Item ID
   ↓                ↓
[User Emb]      [Item Emb]
   ↓                ↓
   └─── GMF ────────┘  (Generalized Matrix Factorization)
   │                │
   └─── MLP ────────┘  (Multi-Layer Perceptron)
          ↓
      [Fusion]
          ↓
    [Prediction]
```

### Python Implementation

```python
import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32, mlp_layers=[64, 32, 16]):
        super().__init__()
        
        # Embeddings for GMF (Generalized Matrix Factorization)
        self.gmf_user_emb = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embedding_dim)
        
        # Embeddings for MLP
        self.mlp_user_emb = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_modules = []
        in_dim = 2 * embedding_dim
        for hidden_dim in mlp_layers:
            mlp_modules.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Fusion: concatenate GMF and MLP outputs
        self.output_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.gmf_user_emb.weight, std=0.01)
        nn.init.normal_(self.gmf_item_emb.weight, std=0.01)
        nn.init.normal_(self.mlp_user_emb.weight, std=0.01)
        nn.init.normal_(self.mlp_item_emb.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        # GMF part: element-wise product of user and item embeddings
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_output = gmf_user * gmf_item  # (batch_size, embedding_dim)
        
        # MLP part: concatenate user and item embeddings
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)  # (batch_size, 2*embedding_dim)
        mlp_output = self.mlp(mlp_input)  # (batch_size, mlp_layers[-1])
        
        # Fusion
        fusion = torch.cat([gmf_output, mlp_output], dim=1)
        logit = self.output_layer(fusion)
        
        return torch.sigmoid(logit.squeeze())

# Training with BPR loss (Bayesian Personalized Ranking)
class BPRLoss(nn.Module):
    def forward(self, pos_scores, neg_scores):
        """
        pos_scores: scores for positive items (user interacted)
        neg_scores: scores for negative items (user didn't interact)
        """
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

# Training example
n_users = 1000
n_items = 500
batch_size = 256

model = NeuralCollaborativeFiltering(n_users, n_items, embedding_dim=32)
criterion = BPRLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Simulated data: (user, positive_item, negative_item) triplets
user_ids = torch.randint(0, n_users, (batch_size,))
pos_item_ids = torch.randint(0, n_items, (batch_size,))
neg_item_ids = torch.randint(0, n_items, (batch_size,))

# Training step
model.train()
optimizer.zero_grad()

pos_scores = model(user_ids, pos_item_ids)
neg_scores = model(user_ids, neg_item_ids)
loss = criterion(pos_scores, neg_scores)

loss.backward()
optimizer.step()

print(f"NCF Training loss: {loss.item():.4f}")

# Inference: Recommend top-k items for a user
model.eval()
with torch.no_grad():
    test_user = torch.tensor([0]).repeat(n_items)  # Replicate user for all items
    all_items = torch.arange(n_items)
    
    scores = model(test_user, all_items)
    top_k = 10
    top_scores, top_indices = torch.topk(scores, k=top_k)
    
    print(f"\nTop-{top_k} recommendations for user 0:")
    for rank, (item_id, score) in enumerate(zip(top_indices, top_scores), 1):
        print(f"  {rank}. Item {item_id.item()}: score {score.item():.4f}")
```

### Variants
- **GMF**: Generalized Matrix Factorization (linear interaction)
- **MLP**: Multi-layer perceptron (nonlinear)
- **NeuMF**: Fusion of GMF + MLP (best performance)

### When NOT to Use
- Very sparse data → Use matrix factorization (more robust)
- Need real-time updates → Requires retraining
- Interpretability critical → Use item-item CF

### Real-World Applications
- **Netflix**: Neural recommendation models
- **Spotify**: Playlist continuation
- **E-commerce**: Product recommendations

---

## 27. Multi-Task Learning for RecSys (MMoE)

### When to Use
- **Multiple objectives**: CTR + conversion, watch time + likes, clicks + shares
- Shared user/item representations across tasks
- Improve performance on sparse tasks via knowledge transfer
- Ranking stage in recommendation pipeline

### Pros
- **Task synergy**: Main task benefits from auxiliary tasks
- **Parameter efficiency**: Shared bottom layers reduce parameters
- **Handles task conflicts**: MMoE learns task-specific expert weights
- **Production-ready**: Used at Google, Alibaba, ByteDance

### Cons
- **Complex training**: Multiple loss functions to balance
- **Task weighting**: Need to tune loss weights carefully
- **Negative transfer**: Unrelated tasks can hurt performance

### Architecture (MMoE)

```
                   Input Features
                         ↓
              [Shared Embedding]
                         ↓
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
    [Expert 1]      [Expert 2]      [Expert 3]
         ↓               ↓               ↓
         └───────────────┼───────────────┘
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
        [Gate Network          [Gate Network
         for Task 1]            for Task 2]
              ↓                     ↓
        [Task 1 Tower]        [Task 2 Tower]
              ↓                     ↓
        CTR Prediction      Conversion Prediction
```

### Python Implementation

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class MMoE(nn.Module):
    def __init__(self, input_dim, num_experts=3, expert_hidden_dim=128, 
                 tower_hidden_dim=64, num_tasks=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # Shared experts
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim) for _ in range(num_experts)
        ])
        
        # Gate networks (one per task)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])
        
        # Task-specific towers
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_dim, tower_hidden_dim),
                nn.ReLU(),
                nn.Linear(tower_hidden_dim, 1)
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, x):
        # Get expert outputs: (batch_size, num_experts, expert_hidden_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        task_outputs = []
        for task_id in range(self.num_tasks):
            # Gate weights for this task: (batch_size, num_experts)
            gate_weights = self.gates[task_id](x)
            
            # Weighted combination of experts: (batch_size, expert_hidden_dim)
            weighted_expert = torch.sum(
                gate_weights.unsqueeze(2) * expert_outputs, dim=1
            )
            
            # Task-specific tower
            task_output = self.towers[task_id](weighted_expert)
            task_outputs.append(torch.sigmoid(task_output.squeeze()))
        
        return task_outputs

# Training example
input_dim = 100  # Feature dimension
batch_size = 256

model = MMoE(
    input_dim=input_dim,
    num_experts=3,
    expert_hidden_dim=128,
    tower_hidden_dim=64,
    num_tasks=2  # Task 1: CTR, Task 2: Conversion
)

# Simulated data
x = torch.randn(batch_size, input_dim)
y_ctr = torch.randint(0, 2, (batch_size,)).float()  # CTR labels
y_cvr = torch.randint(0, 2, (batch_size,)).float()  # Conversion labels

# Training
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
optimizer.zero_grad()

# Forward pass
ctr_pred, cvr_pred = model(x)

# Multi-task loss (weighted sum)
loss_ctr = criterion(ctr_pred, y_ctr)
loss_cvr = criterion(cvr_pred, y_cvr)
loss = 0.7 * loss_ctr + 0.3 * loss_cvr  # Weight main task (CTR) higher

loss.backward()
optimizer.step()

print(f"MMoE Loss - CTR: {loss_ctr.item():.4f}, CVR: {loss_cvr.item():.4f}, Total: {loss.item():.4f}")

# Inference
model.eval()
with torch.no_grad():
    test_x = torch.randn(10, input_dim)
    ctr_pred, cvr_pred = model(test_x)
    print(f"\nPredictions (first 5):")
    for i in range(5):
        print(f"  Sample {i}: CTR={ctr_pred[i]:.4f}, CVR={cvr_pred[i]:.4f}")
```

### Real-World Applications
- **YouTube**: Watch time + engagement (likes, shares)
- **Alibaba**: Click + purchase conversion
- **Google Ads**: CTR + conversion rate
- **TikTok**: Watch time + likes + shares

### Advanced Variants
- **PLE (Progressive Layered Extraction)**: Better handles negative transfer
- **ESMM (Entire Space Multi-Task Model)**: Addresses sample selection bias (CVR = CTR × CTCVR)

---

## Key Trends (2024-2026)

1. **Mixture of Experts (MoE)**: Sparse activation for efficiency (Mixtral, GPT-4)
2. **Long Context**: 1M+ token models (Gemini 1.5, Claude 3)
3. **Multimodal Native**: Text + Image + Audio + Video (GPT-4o, Gemini)
4. **Parameter-Efficient Fine-Tuning**: LoRA, QLoRA dominate
5. **Open Source Catching Up**: Llama 3, Mistral competitive with closed models
6. **Diffusion Speedup**: 1-step generation with consistency models
7. **RAG Everywhere**: Standard for production LLM apps
8. **State Space Models**: Mamba challenges Transformer dominance
9. **Two-Tower + Cross-Attention**: Hybrid retrieval + ranking (RecSys)
10. **Multi-Task Learning**: MMoE/PLE for multi-objective optimization

---

**Last Updated**: 2026-01-27  
**Version**: 1.1 (added recommender systems algorithms)  
**Maintainer**: Data Science Team

**Note**: This field evolves rapidly. Check arXiv, Hugging Face, and Papers with Code for latest developments.
