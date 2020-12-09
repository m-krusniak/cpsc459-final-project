


# Usage

### Train a model

`train_inference`

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
All of our data is taken from the Expanded Groove MIDI Dataset, available [here](https://magenta.tensorflow.org/datasets/e-gmd). Only the subsets of data featured in the paper are included in this repository; feel free to test it on other parts of the Expanded Groove MIDI Dataset to test its performance.

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