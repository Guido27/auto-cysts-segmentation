## Unet++

| Optimizer | Scheduler                  | Loss          | LR     | Batch size |
|:--------- |:-------------------------- |:------------- |:------ |:---------- |
| Adam      | CosineAnnealingWarmRestart | BCEwithLogits | 0.0001 | 4          |

Baseline.

- MaxEpochs: 100
- EarlyStopping: True
    - Patience: 20   
- Test IoU: 0.5839
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/in4puupd/overview?workspace=user-guidog)

## CE-Net

### Run 1

| Optimizer | Scheduler                  | Loss          | LR     | Batch size |
|:--------- |:-------------------------- |:------------- |:------ |:---------- |
| Adam      | CosineAnnealingWarmRestart | BCEwithLogits | 0.0001 | 4          |

CE-Net con questa configurazione **non imparava**. La loss era molto bassa ma l'output (durante training e validation) era sempre un' immagine completamente nera, ciò è dovuto probabilmente alla struttura delle groud truth: essendo prevalentemente nere con una porzione di foreground molto piccola rispetto al background, il modello imparava che una predizione in cui tutti i pixel sono classificati come "background" fosse una buona predizione. A questo si aggiunge il tipo di loss usata, probabilmente non adatta al modello in questione. 

- MaxEpochs: 100
- EarlyStopping: True
    - Patience: 20
- Test IoU: 0


**Aggiornamento**: Allenando il modello con 50 epoche (senza earlystopping) i risultati migliorano, e di molto, rispetto a quanto detto sopra. Durante il training si osserva che la metrica "train_iou" passa da un valore di 0 delle prime 13 epoche a 0.56 nell'epoca 14, migliora fino ad arrivare all'epoca 44 che restituisce il miglior risultato sul validation set in termini di IoU. Il test restituisce buoni risultati.

- MaxEpochs: 50
- EarlyStopping: False
- Test IoU: 0.5891
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/y2360dg6/overview?workspace=user-guidog)

### Run 2

| Optimizer | Scheduler                  | Loss     | LR     | Batch size |
| --------- | -------------------------- | -------- |:------:|:----------:|
| Adam      | CosineAnnealingWarmRestart | DiceLoss | 0.0001 | 4          |

Cambiando la loss e usando la DiceLoss il modello sembra imparare prima cosa segmentare e gli output durante training e test sono sensati fin da subito. Probabilmente CE-Net lavora bene con la Dice, che è proprio la loss usata dagli autori del paper, nello specifico gli autori introducono una versione con un termine per effettuare regularization (la versione usata qui non è regolarizzata).

- MaxEpochs 100
- EarlyStopping: True
    - Patience: 20
- Test IoU: 0.6124
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/exm43q8b/overview?workspace=user-guidog) 


### Note
Implementazione Dice Loss utilizzata: [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)
