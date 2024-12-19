import sentencepiece as spm
import torch
from transformer_model import FridayTransformer


def predict(model, src_sentence, tokenizer, max_len=500):
    model.eval()  # Set the model to evaluation mode
    # Tokenize and convert the src_sentence to a tensor
    src_indexes = tokenizer.Encode(src_sentence)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0)  # Add batch dimension
    # Initialize the target sequence with the start token
    trg_indexes = [tokenizer.bos_id()]
    # Loop to generate each word in the target sequence
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            preds = model(src_tensor, trg_tensor)
        # Get the index of the highest probability token
        pred_token = preds.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        # Stop if the model predicts the end of sentence token
        if pred_token == tokenizer.eos_id():
            break
    # Convert the predicted indexes back to words
    return trg_indexes


# Load the tokenizer
tokenizer_path = 'models/tokenizer/model.model'
sp_model = spm.SentencePieceProcessor()
sp_model.Load(tokenizer_path)

# Initialize the Transformer model
embed_dim = 512
vocab_size = sp_model.vocab_size()
num_layers = 2
n_heads = 8
model = FridayTransformer(embed_dim=embed_dim,
                    src_vocab_size=vocab_size,
                    target_vocab_size=vocab_size,
                    num_layers=num_layers,
                    n_heads=n_heads)

# Load the trained model weights
checkpoint = torch.load('friday-torch.pth', weights_only=True)
model.load_state_dict(checkpoint)

# Apply quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Convert to TorchScript and optimize for inference
model = torch.jit.script(model)
model = torch.jit.optimize_for_inference(model)

# Ask the user for input
src_sentence = input("Enter the source sentence: ")

# Predict the output
predicted_sentence = predict(model, src_sentence, sp_model)

# Print the predicted sentence
print(sp_model.Decode(predicted_sentence))
