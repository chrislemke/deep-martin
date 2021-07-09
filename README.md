<img src="https://github.com/stoffy/martin/blob/master/images/title_image.png?raw=true" alt="Deep Martin">

<h1>Deep Martin - text simplification</h1>
<h2>the Heidegger countermovement</h2>
<blockquote><a href="https://www.deepl.com/translator#de/en/Danach%20ist%20das%20In-der-Welt-sein%20ein%20Sich-vorweg-schon%20sein-in-der%20Welt%20als%20Sein-bei-innerweltlich-begegnendem-Seienden">Danach ist das In-der-Welt-sein ein Sich-vorweg-schon sein-in-der Welt als Sein-bei-innerweltlich-begegnendem-Seienden</a><br><b>Martin Heidegger</b></blockquote>
<h3>Unsimplifiable, untranslatable</h3>

<p>Language as a fundamental characteristic of man and society is the center of NLP. It has the potential of great enlightenment, as well as great concealment.  Language and thinking must be brought into harmony.
Simplification of language leads to the democratization of knowledge. Thus, it can provide access to knowledge that may otherwise be hidden. No more complex language!
<br><br>
Deep Martin aims to contribute to this.
The project is dedicated to different models to make complicated and complex content accessible to all. 
It follows the approach of <a href="https://simple.wikipedia.org/wiki/Main_Page">Simple Wikipedia</a>.</p> 

<h3>About the project</h3>

<h3>How to use</h3>

<p>Two different approaches are available. 
One is to use the super nice <a href="https://huggingface.co">Hugging Face</a> library. 
This can be used to create various state of the art sequence to sequence models. 
The other part is a self-made transformer. 
Here it is mainly about trying out different approaches.</p>
<h4>Hugging Face</h4>
<p>For using the Hugging Face implementation you need to provide a dataset. It needs to have one column with the normal version (<code>Normal</code>)
and one for the simplified version (<code>Simple</code>).
The <code>HuggingFaceDataset</code> class can help you with it.<br>To train
a model you then simple run something like:<br></p>
<pre>
python /your/path/to/deep-martin/src/hf_transformer_trainer.py \
--eval_steps 5000 \ # This number should be based on the size of the dataset. 
--warmup_steps 800 \ # This number should be based on the size of the dataset.
--ds_path /path/ \ # Path to you dataset.
--save_model_path /path/ \ # Path to where the trained model should be stored.
--training_output_path /path/ \ # Path to where the checkpoints and the training data should be stored.
--tokenizer_id bert-base-cased # Path or identifier to Hugging Face tokenizer.
</pre>
<p>There are a lot more parameters. Check out <code>hf_transformer_trainer.py</code> to get an overview.</p>

<h4>Self-made-transformer</h4>
<p>This transformer is more for experimenting. Have a look at the code and get an overview about what is going on.
To train the self-made-transformer a train and a test dataset as CSV is needed. This will be transformed
to a suitable dataset at the beginning of the training. Same as with the transformers from above the dataset needs to have one column with the normal version (<code>Normal</code>)
and one for the simplified version (<code>Simple</code>) <br>
To start the training you can run:<br></p>
<pre>
python /your/path/to/deep-martin/src/custom_transformer_trainer.py \
--ds_path /path \ # Path of the folder which contains the `train_file.csv` and the `test_file.csv`
--train_file train_file.csv \
--test_file test_file.csv \
--epochs 3 \
--save_model_path /path/ # Path to where the trained model should be stored.
</pre>

<h3>Challenges</h3>
<p>Let's talk about the problems in this project. </p>
<h4>Dataset</h4>
<p>As is so often the case, one problem lies in obtaining high-quality data.
Multiple datasets where used for this project. You can find them 
<a href="https://paperswithcode.com/task/text-simplification">here</a>.<br>
While the ASSET dataset provides a very good quality due to the multiple simplification of each record, its size is simply too small for training a transformer. 
This problem is also true for other datasets. 
The two datasets based on Wikipedia unfortunately suffer from 
lack of quality. Either records are not simplification, 
but simply the same article. Or the simplification is of poor quality. In both cases, using it meant worse results.
To increase the overall quality, the records were compared and 
filtered out using <a href="https://radimrehurek.com/gensim/models/doc2vec.html">Doc2Vec</a> and cosine distance. 
</p>

<h4>Model size and computation</h4>
<p>
Transformers are huge, need a lot of data and a lot of time to train. 
<a href="http://research.google.com/colaboratory/">Google colab</a> can help, but it is not the most convenient way. 
With the help of <a href="https://aws.amazon.com/de/ec2">AWS EC2</a>, things can be sped up a lot and training of larger models is also possible.
</p>

<h3>Next steps</h3>
<p>Since the self-made-transformer is a work in progress project it is never finished.
It is made for learning and trying out. One interesting idea is to use the
transformer as a generator in a GAN to improve to overall output.</p>


