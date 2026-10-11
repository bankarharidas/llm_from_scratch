# import torch
# from GPT import MiniGPT

# from tokenizer import vocab_size, data, model, train_data
# import torch.nn.functional as F


# def main():
#     # =========================
#     # Training
#     # =========================

#     model.train()

#     best_val_loss = float("inf")

#     for step in range(num_steps):

#         # Get training batch
#         inputs, targets = get_batch(train_data, batch_size, sequence_length)

#         # Forward pass
#         logits, _ = model(inputs)

#         # Loss
#         loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

#         # Backpropagation
#         optimizer.zero_grad()

#         loss.backward()

#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#         # Update learning rate

#         current_lr = get_learning_rate(step)

#         for param_group in optimizer.param_groups:
#             param_group["lr"] = current_lr

#         # Update parameters
#         optimizer.step()

#         # Training loss
#         if step % 100 == 0:

#             print(
#                 f"Step {step}: " f"Train Loss {loss.item():.4f} " f"LR {current_lr:.6f}"
#             )

#         # =========================
#         # Validation
#         # =========================

#         if step % 500 == 0:

#             model.eval()

#             with torch.no_grad():

#                 val_inputs, val_targets = get_batch(
#                     val_data, batch_size, sequence_length
#                 )

#                 val_logits, _ = model(val_inputs)

#                 val_loss = F.cross_entropy(
#                     val_logits.reshape(-1, vocab_size), val_targets.reshape(-1)
#                 )

#             val_loss_value = val_loss.item()

#             print(f"Step {step}: " f"Validation Loss {val_loss_value:.4f}")

#             # =========================
#             # Save ONLY if better
#             # =========================

#             if val_loss_value < best_val_loss:

#                 best_val_loss = val_loss_value

#                 best_checkpoint = {
#                     "model_state_dict": model.state_dict(),
#                     "optimizer_state_dict": optimizer.state_dict(),
#                     "step": step,
#                     # Model configuration
#                     "vocab_size": vocab_size,
#                     "embedding_size": embedding_size,
#                     "num_heads": num_heads,
#                     "num_blocks": num_blocks,
#                     "sequence_length": sequence_length,
#                     "ffn_hidden_size": ffn_hidden_size,
#                     # BPE tokenizer
#                     "tokenizer_file": "tokenizer.json",
#                 }

#                 torch.save(best_checkpoint, "best_model_kv.pth")

#                 print("Best model saved!")

#             model.train()
