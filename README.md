

# Dependencies
This project requires the following:

* **Python 2**, of course
* **TensorFlow**, for creating and using models
* **[MIDO](https://mido.readthedocs.io/en/latest/)**(`pip install mido`), a Python package for manipulating MIDI data
* **[scikit-learn](https://scikit-learn.org/stable/)**, for random forest-backed inference machines

If you'd like to visualize, hear, and combine MIDI outputs, there's a variety of software capable of parsing a score from MIDI data. We've used Musescore for the visualizations, which is free software; Sibelius is a professional alternative if you've worked with scores before.

You can download the **[Extended MIDI Groove Dataset]([here](https://magenta.tensorflow.org/datasets/e-gmd)** for data with which to play.

# Usage
This project operates around MIDI files as a middleman representing a user's performance and what another user might hear. In an end-to-end system, the MIDI data would be generated live by the performer(s) when the model is in use. However, for our purposes it is more convenient to represent all data as static MIDI files. (Due to the nature of MIDI this makes no difference from the perspective of the model but makes testing much easier for anyone without access to a MIDI instrument.)


### Train a model
For accurate predictions, it's important to use a model that was trained on music with the same stylistic tendencies as the music you'd like to predict. The syntax is:

`make_model.py <model_type> <train_dir> <model_file>`

where
* `model_type` is one of `ff`, `im_mlp`, or `im_rf`, to train a simple feed-forward network, a multi-layer-perceptron-backed inference machine, or a random-forest-backed inference machine,
* `train_dir` is a directory full of MIDI files on which to train the model, and
* `model_file` is the file to which you would like to save the trained model.

The trained model can then be used to make predictions on new MIDI files.

### Generate MIDI predictions from an input

`generate_midi <model_type> <model_file> <input_dir> <output_dir>`

### Evaluate a model's performance


### Evaluate the difference between genres
`genre_analysis` will do this for you:

`./genre_analysis.py <genres-dir>`

The analysis will train each available models on the data in every folder in `genres-dir/train` and evaluate them on the data in the corresponding folders in `genres-dir/test`. For readability, the output of the analysis is placed in `genre_output.txt`. 


### Examine output from specific MIDI files
Sometimes one may want to compare the true rhythm and the rhythm output by the models as MIDI. You can use `generate_midi.py` to do this:

 
# Data

### Bulk data
All of our data is taken from the Expanded Groove MIDI Dataset, available [here](https://magenta.tensorflow.org/datasets/e-gmd). Only the subsets of data featured in the paper are included in this repository; feel free to use it on other parts of the dataset to test its performance.

### Genres
We separate a portion of the E-GMD data into four distinct genres: "rock," "funk," "jazz," and "latin" (including afrocuban, reggae, and reggaeton), then further divided them into testing and training sets. These can be viewed in the `data/genre/` directory.

### Standard tests
Additionally, we developed a set of seven rhythms from simple to complex in order to test the specific behavior of the models. Each is sixteen measures (256 frames) long. They are available in `data/standard/` and include:

* Standard 1: Steady quarter notes
* Standard 2: Steady eighth notes
* Standard 3: Steady sixteenth notes
* Standard 4: "One-and-a" beat repeating each measure
* Standard 5: A repeated pattern with no downbeat
* Standard 6: "Double Beat", a realistic warm-up with few repititions
* Standard 7: A simple rock beat with different instruments.