import torch
import sentencepiece as spm
import traceback
import warnings
import sys


# Definisci una funzione per catturare i warning
def catch_warnings():
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Temporaneamente riabilita i warning per poterli catturare
        warnings.resetwarnings()
        warnings.simplefilter("always")

        try:
            # Esegui il codice che genera warning
            create_and_test_model()
        except Exception as e:
            print(f"Si è verificata un'eccezione: {e}")
            traceback.print_exc()

        # Stampa tutti i warning catturati con i loro stack traces
        if caught_warnings:
            print("\n===== WARNING CATTURATI =====")
            for i, w in enumerate(caught_warnings):
                print(f"\nWarning {i + 1}:")
                print(f"  Messaggio: {w.message}")
                print(f"  Categoria: {w.category.__name__}")
                print(f"  File: {w.filename}:{w.lineno}")
                print("\nStack trace:")
                traceback.print_stack(w.__traceback__.tb_frame)
                print("=" * 50)


def create_and_test_model():
    # Definisci il dispositivo (CUDA o CPU)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {DEVICE}")

    # Stampa informazioni su PyTorch
    print(f"Versione PyTorch: {torch.__version__}")

    # Carica il modello SentencePiece
    try:
        tokenizer_path = "models/tokenizer/model.model"
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(tokenizer_path)
        print(f"Tokenizer caricato correttamente. Dimensione vocabolario: {sp_model.GetPieceSize()}")
    except Exception as e:
        print(f"Errore nel caricamento del tokenizer: {e}")
        return

    # Parametri del modello
    embed_dim = 512
    src_vocab_size = sp_model.GetPieceSize()
    target_vocab_size = src_vocab_size
    num_layers = 6
    d_ff = 2048
    n_heads = 8
    dropout = 0.1

    print("Creazione modello SimplifiedTransformer...")

    # Importa qui la classe per isolarne l'importazione
    from transformer_model import SimplifiedTransformer

    # Crea l'istanza del modello
    model = SimplifiedTransformer(embed_dim, src_vocab_size, target_vocab_size, num_layers, d_ff, n_heads, dropout)
    print("Modello creato correttamente")

    # Testa il modello con input di esempio
    print("\nTest del modello con input di esempio...")

    # Crea input di esempio
    src = torch.randint(1, src_vocab_size, (2, 10))  # batch_size=2, seq_len=10
    tgt = torch.randint(1, target_vocab_size, (2, 8))  # batch_size=2, seq_len=8

    print(f"Input src shape: {src.shape}")
    print(f"Input tgt shape: {tgt.shape}")

    # Forward pass
    print("Esecuzione forward pass...")
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")

    # Test encoder
    print("\nTest dell'encoder...")
    memory = model.encode(src)
    print(f"Memory shape: {memory.shape}")

    # Test decoder
    print("\nTest del decoder...")
    decoder_output = model.decode(tgt, memory)
    print(f"Decoder output shape: {decoder_output.shape}")

    # Test generazione
    print("\nTest della generazione...")
    generated = model.generate(src, sp_model.bos_id(), max_len=20)
    print(f"Generated shape: {generated.shape}")

    print("\nTutti i test completati con successo!")


if __name__ == "__main__":
    catch_warnings()