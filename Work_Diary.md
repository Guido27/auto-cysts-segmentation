## Info
Vengonop riportate le run dei modelli usando la baseline come riferimento quindi NON con multiscale 
## Unet++

| Optimizer | Scheduler                  | Loss          | LR     | Batch size |
|:---------: |:--------------------------: |:-------------: |:------: |:----------: |
| Adam      | CosineAnnealingWarmRestart | BCEwithLogits | 0.0001 | 4          |

Baseline. Eseguita 2 volte, la prima senza evaluator (per sbaglio) e la seconda con evaluator, risultati differenti (?)

- MaxEpochs: 100
- EarlyStopping: True
    - Patience: 20   
- Test IoU: 0.5839/0.5393
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/in4puupd/overview?workspace=user-guidog)

## CE-Net

### Run 1

| Optimizer | Scheduler                  | Loss          | LR     | Batch size |
|:---------: |:--------------------------: |:-------------: |:------: |:----------: |
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

| Optimizer | Scheduler                      | Loss       | LR     | Batch size       |
|:---------:| :----------------------------: | :--------: |:------:|:----------------:|
| Adam      | CosineAnnealingWarmRestart     | DiceLoss   | 0.0001 | 4                |

Cambiando la loss e usando la DiceLoss il modello sembra imparare prima cosa segmentare e gli output durante training e test sono sensati fin da subito. Probabilmente CE-Net lavora bene con la Dice, che è proprio la loss usata dagli autori del paper, nello specifico gli autori introducono una versione con un termine per effettuare regularization (la versione usata qui non è regolarizzata).

- MaxEpochs 100
- EarlyStopping: True
    - Patience: 20
- Test IoU: 0.6124
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/exm43q8b/overview?workspace=user-guidog) 

### Run 3

| Optimizer | Scheduler                  | Loss     | LR     | Batch size |
| :---------: | :--------------------------: | :--------: |:------:|:----------:|
| SGD       | CosineAnnealingWarmRestart | DiceLoss | 0.004  | 4          |

Il paper di CE-Net usa SGD quindi in questa run ho usato tale Optimizer piuttosto che Adam. I risultati non sembrano migliori, **si potrebbe provare ad eseguire una run con gli stessi parametri ma senza EarlyStopping e magari molte epoche** per vedere se effettivamente SGD porta ad una soluzione migliore.

- MaxEpochs 50
- EarlyStopping: True 
    - Patience: 10
- Test IoU: 0.5443
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/mbokjz26/overview?workspace=user-guidog)

### Run 4

| Optimizer | Scheduler                  | Loss     | LR     | Batch size |
| :---------: |:--------------------------: | :--------: | :--------: | :------------:|
| SGD       | PolynomialLR               | DiceLoss | 0.004  | 4          |

In questa run ho simulato (il più possibile) la configurazione presentata nel paper di CE-Net, la differenza è che loro usano un PolynomialLR che agisce in 3 step al termine di determinate epoche, ad esempio epoche 10,40 e 70 mentre quello usato qui è l'implementazione di Pytorch che modifica il LR secondo il polinomio nelle prime *X* epoche, dove *X* è il numero di step specificati. **Si potrebbe provare ad usare il PolynomialLR uguale a quello del paper per confrontare**.

- MaxEpochs 50
- EarlyStopping: True 
    - Patience: 10
- Test IoU: 0.5173
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/6nvlxdq5/overview?workspace=user-guidog)

### Run 5
| Optimizer  | Scheduler                   | Loss         | LR    | Batch size |
| :--------: | :-------------------------: | :----------: |:-----:|:----------:|
| Adam       | CosineAnnealingWarmRestart  | BCEDiceLoss  | 0.0001 | 4          |

Test di CE-Net con loss BCEDice, una combinazione di BCE e Dice (con peso rispettivamente settato a 1 e 1). 2 test effettuati, il primo con EarlyStopping e il secondo senza, risultati simili. **Si potrebbe provare a settare i pesi in maniera differente**

- MaxEpochs 50
- EarlyStopping: True prima run, False seconda 
    - Patience: 10
- Test IoU: 0.572 senza earlystop, 587 con earlystop
- Run con EarlyStopping: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/70dasq0m/overview?workspace=user-guidog)
- Run senza EarlyStopping: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/gprm89np/overview?workspace=user-guidog)
### Note
Implementazione Dice Loss e BCEDice Loss utilizzate prese da: [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)

## CaraNet
### Run 1
| Optimizer  | Scheduler                   | Loss         | LR    | Batch size |
| :--------: | :-------------------------: | :----------: |:-----:|:----------:|
| Adam       | CosineAnnealingWarmRestart  | structure_loss  | 0.0001 | 4          |

Prima run effettuata con CaraNet usando le augmentation del progetto di base, nel paper non usano augmentation ma multi-scale train strategy
La loss usata è quella del paper, ovvero una combinazione lineare di BCE e IoU.

-MaxEpochs 100
- EarlyStopping True
    - Patience: 10
- Test Iou: 0.5936
- Run: [Weights & Biases](https://wandb.ai/guidowandb/rene-policistico-cyst_segmentation/runs/fjqzh6oo/overview?workspace=user-guidog)
