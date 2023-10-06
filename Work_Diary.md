

## Unet++

Baseline per confronto

- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/ape1j3lj/overview?workspace=user-guidog)

## CE-Net

### Configuration A

| Optimizer | Loss            | LR     | Batch size |
| --------- | --------------- |:------:|:----------:|
| Adam      | BCE with Logits | 0.0001 | 4          |

CE-Net con questa configurazione **non imparava**. La loss era molto bassa ma l'output (durante training e validation) era sempre un' immagine completamente nera, ciò è dovuto probabilmente alla struttura delle groud truth: essendo prevalentemente nere con una porzione di foreground molto piccola rispetto al background, il modello imparava che una predizione in cui tutti i pixel sono classificati come "background" fosse una buona predizione. A questo si aggiunge il tipo di loss usata, probabilmente non adatta al modello in questione. 

- Test IoU: 0

### Configuration B

| Optimizer | Loss     | LR     | Batch size |
| --------- | -------- |:------:|:----------:|
| Adam      | DiceLoss | 0.0001 | 4          |

Cambiando la loss e usando la DiceLoss il modello sembra imparare correttamente cosa segmentare e gli output durante training e test sono sensati. Probabilmente CE-Net lavora bene con la Dice, che è proprio la loss usata dagli autori del paper, nello specifico gli autori introducono una versione con un termine per effettuare regularization (la versione usata qui non è regolarizzata)

- Test IoU: 0.5873
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/0ni79nnm/overview?workspace=user-guidog)
- Implementazione Dice Loss: [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)

### 
