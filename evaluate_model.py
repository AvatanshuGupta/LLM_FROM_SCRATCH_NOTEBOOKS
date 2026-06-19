import torch
import tiktoken
import torch.nn as nn


# ==========================================
# UTILS
# ==========================================

def text_to_token_ids(text,tokenizer):
    tokenized_text=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    text_tensor=torch.tensor(tokenized_text).unsqueeze(0)
    return text_tensor

def token_ids_to_text(ids,tokenizer):
    ids=ids.squeeze(0)
    return tokenizer.decode(ids.tolist())

# ==========================================
# ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_biasing=False):
        super().__init__()
        assert (d_out % num_heads ==0), \
        "d_out must be divisible by num_heads"

        self.d_out=d_out
        self.num_heads=num_heads

        self.head_dim=d_out // num_heads

        self.W_query=nn.Linear(d_in,d_out,bias=qkv_biasing)
        self.W_key=nn.Linear(d_in,d_out,bias=qkv_biasing)
        self.W_value=nn.Linear(d_in,d_out,bias=qkv_biasing)
        self.out_proj=nn.Linear(d_out,d_out)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
                        "mask",
                        torch.triu(torch.ones(context_length,context_length),diagonal=1)
                             )
        
    def forward(self,x):
        b,num_tokens,d_in=x.shape

        keys=self.W_key(x)
        values=self.W_value(x)
        queries=self.W_query(x)

        keys=keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values=values.view(b,num_tokens,self.num_heads,self.head_dim)
        queries=queries.view(b,num_tokens,self.num_heads,self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores=queries @ keys.transpose(2,3)

        mask_bool=self.mask.bool()[:num_tokens,:num_tokens]

        attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        attn_weights=self.dropout(attn_weights)

        context_vector=(attn_weights @ values).transpose(1,2)

        context_vector=context_vector.contiguous().view(b,num_tokens,self.d_out)
        context_vector=self.out_proj(context_vector)

        return context_vector


class LayerNorm(nn.Module):

    def __init__(self,embedding_dim):
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(embedding_dim))
        self.shift=nn.Parameter(torch.zeros(embedding_dim))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x=(x-mean)/torch.sqrt(var+self.eps)

        return self.scale*norm_x + self.shift



class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi))*( x + 0.044715 * torch.pow(x,3) )
            ))
    

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Linear(cfg['emb_dim'],4 * cfg['emb_dim'] ),
            GELU(),
            nn.Linear( 4 * cfg['emb_dim'],cfg['emb_dim'])
            )
        
    def forward(self,x):
        return self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn=MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_biasing=cfg["qkv_bias"]
        )
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg['emb_dim'])
        self.norm2=LayerNorm(cfg['emb_dim'])
        self.drop_shortcut=nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut=x
        x=self.norm1(x)
        x=self.attn(x)
        x=self.drop_shortcut(x)
        x= x + shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x= x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb=nn.Dropout(cfg["drop_rate"])

        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm=LayerNorm(cfg["emb_dim"])
        self.out_head=nn.Linear(
            cfg["emb_dim"],cfg["vocab_size"],bias=False
        )

    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        tok_emb=self.tok_emb(in_idx)
        pos_emb=self.pos_emb(torch.arange(seq_len,device=in_idx.device))

        x=tok_emb + pos_emb
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits
    
def generate(model,idx,max_next_tokens,context_size,temperature=0.0,top_k=None,eos_id=None):
    for _ in range(max_next_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        logits=logits[:,-1,:]

        if top_k is not None:
            top_logits,top_pos=torch.topk(logits,top_k)
            min_value=top_logits[:,-1]
            logits=torch.where(
                condition=logits < min_value,
                input=torch.tensor(float("-inf")).to(logits.device),
                other=logits
            )

        if temperature > 0.0 :
            logits=logits/temperature
            probas=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probas,num_samples=1)
        
        else :
            idx_next=torch.argmax(logits,dim=-1,keepdim=True)
        
        if eos_id is not None and (idx_next == eos_id).any():
            break

        idx=torch.cat((idx,idx_next),dim=1)
    
    return idx

# ============================================================
# CONFIG
# ============================================================

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 640,
    "n_heads": 10,      # likely
    "n_layers": 10,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
MODEL_PATH = input("Enter model checkpoint path: ").strip()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

state_dict = checkpoint["model_state_dict"]

for k, v in state_dict.items():
    print(k, v.shape)
    break

print(f"\nLoading model from: {MODEL_PATH}")

model = GPTModel(GPT_CONFIG)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handles both:
# torch.save(model.state_dict())
# torch.save({"model_state_dict": ...})

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("✓ Model loaded successfully")

tokenizer = tiktoken.get_encoding("gpt2")

# ============================================================
# GENERATION FUNCTION
# ============================================================

def run_prompt(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
):
    encoded = text_to_token_ids(prompt, tokenizer).to(DEVICE)

    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=encoded,
            max_next_tokens=max_new_tokens,
            context_size=GPT_CONFIG["context_length"],
            temperature=temperature,
            top_k=top_k
        )

    generated_text = token_ids_to_text(output_ids, tokenizer)

    print("\n" + "="*80)
    print("PROMPT:")
    print(prompt)
    print("\nOUTPUT:")
    print(generated_text)
    print("="*80)


# ============================================================
# TEST SUITE
# ============================================================

test_cases = {

    "GENERAL KNOWLEDGE": [
        "What is the capital of France?",
        "Who developed the theory of relativity?",
        "Why is the sky blue?",
        "What is photosynthesis?"
    ],

    "REASONING": [
        "If all cats are animals and some animals are black, are all cats black? Explain.",
        "A train travels 60 km in 1 hour. How far will it travel in 3.5 hours?",
        "Tom is taller than Jack. Jack is taller than Sam. Who is tallest?",
        "What comes next: 2, 4, 8, 16, ?"
    ],

    "MATH": [
        "12 + 35 =",
        "15 * 7 =",
        "What is the square root of 144?",
        "Solve: 5x + 10 = 35"
    ],

    "INSTRUCTION FOLLOWING": [
        "List three benefits of exercise.",
        "Write exactly five words about AI.",
        "Give me a recipe for tea in three steps.",
        "Summarize machine learning in one sentence."
    ],

    "CREATIVITY": [
        "Write a short poem about the moon.",
        "Tell a story about a dragon and a scientist.",
        "Describe a city floating in the clouds.",
        "Invent a new animal and describe it."
    ],

    "COMMON SENSE": [
        "Why should you wear a seatbelt?",
        "What would happen if plants disappeared?",
        "Why do people sleep at night?",
        "Why should you drink water?"
    ],

    "CODING": [
        "Write a Python function to add two numbers.",
        "Explain what a for loop is.",
        "What is a variable in programming?",
        "Write a Python hello world program."
    ]
}


# ============================================================
# RUN ALL TESTS
# ============================================================

print("\n")
print("="*80)
print("STARTING MODEL EVALUATION")
print("="*80)

for category, prompts in test_cases.items():

    print("\n")
    print("#"*80)
    print(f"CATEGORY: {category}")
    print("#"*80)

    for prompt in prompts:
        run_prompt(prompt)

print("\nEvaluation completed.")