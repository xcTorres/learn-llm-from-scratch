# Model
The transformer source code is in transformer.py, which is originally from [ethen8181](https://ethen8181.github.io/machine-learning/deep_learning/seq2seq/torch_transformer.html), in which there are detailed explanations of the transformer model.

# Data
Torchtext is not actively maintained, so here huggingface datasets is used. The code is in data.py.

# Train
```
python train.py
```

# Inference
```
python inference.py
```
And you will see the following inference results:
```
source:  Two young, White males are outside near many bushes.
target:  Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
predicted:  Zwei weiße Männer sind im Freien in der Nähe vielen Büschen Büschen.
**************************************************
source:  Several men in hard hats are operating a giant pulley system.
target:  Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.
predicted:  Mehrere Männer mit Schutzhelmen füttern ein riesiges Kissen.
**************************************************
source:  A little girl climbing into a wooden playhouse.
target:  Ein kleines Mädchen klettert in ein Spielhaus aus Holz.
predicted:  Ein kleines Mädchen klettert in einem Holzkonzernen Spielz.
**************************************************
source:  A man in a blue shirt is standing on a ladder cleaning a window.
target:  Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.
predicted:  Ein Mann in blauem Hemd steht auf einer Leiter und putzt eine Fenster.
**************************************************
source:  Two men are at the stove preparing food.
target:  Zwei Männer stehen am Herd und bereiten Essen zu.
predicted:  Zwei Männer bereiten Essen zu und bereiten Essen zu.
**************************************************
```