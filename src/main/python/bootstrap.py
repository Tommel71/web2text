import os
import sys
import time

from sklearn.utils import resample
import pandas as pd

import numpy as np
import tensorflow as tf

from main import evaluate_unary, evaluate_edge
from forward import EDGE_VARIABLES, UNARY_VARIABLES, edge, loss, unary
from viterbi import viterbi
from shuffle_queue import ShuffleQueue
from data import cleaneval, googletrends

BATCH_SIZE = 128
PATCH_SIZE = 9
N_FEATURES = 128
N_EDGE_FEATURES = 25
TRAIN_STEPS = 5000
LEARNING_RATE = 1e-3
DROPOUT_KEEP_PROB = 0.8
REGULARIZATION_STRENGTH = 0.000
EDGE_LAMBDA = 1
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'trained_model_bootstrap_cleaneval_web2text')

TRAIN_SIZE = 55
VAL_SIZE = 5

all_data_cleaneval = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
  27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
  54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,
  85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,
  109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,
  129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,
  149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,
  169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,
  189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,
  209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,
  229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,
  249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,
  269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,
  289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,
  309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,
  329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,
  349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,
  369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,
  389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,
  409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,
  429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,
  449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,
  469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,
  489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,
  509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,
  529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,
  549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,
  569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,
  589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,
  609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,
  629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,
  649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,
  669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,
  689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,
  709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,
  729,730,731,732,733,734,735,736])

all_data_googletrends = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
  27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
  54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,
  85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,
  109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,
  129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,
  149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,
  169,170,171])

def get_batch(q, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE):
  """Takes a batch from a ShuffleQueue of documents"""
  # Define empty matrices for the return data

  batch       = np.zeros((BATCH_SIZE, PATCH_SIZE, 1, N_FEATURES), dtype = np.float32)
  labels      = np.zeros((BATCH_SIZE, PATCH_SIZE, 1, 1), dtype = np.float32)
  edge_batch  = np.zeros((BATCH_SIZE, PATCH_SIZE-1, 1, N_EDGE_FEATURES), dtype = np.float32)
  edge_labels = np.zeros((BATCH_SIZE, PATCH_SIZE-1, 1, 1), dtype = np.int64)


  for entry in range(BATCH_SIZE):
    # Find an entry that is long enough (at least one patch size)
    while True:
      doc = q.takeOne()
      length = doc['data'].shape[0]
      if length > PATCH_SIZE+1:
        break

    # Select a random patch
    i = np.random.random_integers(length-PATCH_SIZE-1)

    # Add it to the tensors
    batch[entry,:,0,:]       = doc['data'][i:i+PATCH_SIZE,:]
    edge_batch[entry,:,0,:]  = doc['edge_data'][i:i+PATCH_SIZE-1,:]
    labels[entry,:,0,0]      = doc['labels'][i:i+PATCH_SIZE] # {0,1}
    edge_labels[entry,:,0,0] = doc['edge_labels'][i:i+PATCH_SIZE-1] # {0,1,2,3} = {00,01,10,11}

  return batch, edge_batch, labels, edge_labels

def test_structured(test_data, lamb=EDGE_LAMBDA):
  unary_features = tf.placeholder(tf.float32)
  edge_features  = tf.placeholder(tf.float32)

  # hack to get the right shape weights
  _ = unary(tf.placeholder(tf.float32, shape=[1,PATCH_SIZE,1,N_FEATURES]), False)
  _ = edge(tf.placeholder(tf.float32, shape=[1,PATCH_SIZE,1,N_EDGE_FEATURES]), False)

  tf.get_variable_scope().reuse_variables()
  unary_logits = unary(unary_features, is_training=False)
  edge_logits  = edge(edge_features, is_training=False)

  unary_saver = tf.train.Saver(tf.get_collection(UNARY_VARIABLES))
  edge_saver  = tf.train.Saver(tf.get_collection(EDGE_VARIABLES))

  init_op = tf.global_variables_initializer()

  with tf.Session() as session:
    session.run(init_op)
    unary_saver.restore(session, os.path.join(CHECKPOINT_DIR, "unary.ckpt"))
    edge_saver.restore(session, os.path.join(CHECKPOINT_DIR, "edge.ckpt"))

    from time import time

    start = time()
    def prediction_structured(features, edge_feat):
      features  = features[np.newaxis, :, np.newaxis, :]
      edge_feat = edge_feat[np.newaxis, :, np.newaxis, :]

      unary_lgts = session.run(unary_logits, feed_dict={unary_features: features})
      edge_lgts = session.run(edge_logits, feed_dict={edge_features: edge_feat})

      return viterbi(unary_lgts.reshape([-1,2]), edge_lgts.reshape([-1,4]), lam=lamb)

    def prediction_unary(features, _):
      features = features[np.newaxis, :, np.newaxis, :]
      logits = session.run(unary_logits, feed_dict={unary_features: features})
      return np.argmax(logits, axis=-1).flatten()

    accuracy, precision, recall, f1 = evaluate_unary(test_data, prediction_structured)
    accuracy_u, precision_u, recall_u, f1_u = evaluate_unary(test_data, prediction_unary)
    end = time()
    print('duration', end-start)
    print('size', len(test_data))
    print("Structured: Accuracy=%.5f, precision=%.5f, recall=%.5f, F1=%.5f" % (accuracy, precision, recall, f1))
    print("Just unary: Accuracy=%.5f, precision=%.5f, recall=%.5f, F1=%.5f" % (accuracy_u, precision_u, recall_u, f1_u))

  return((accuracy, precision, recall, f1), (accuracy_u, precision_u, recall_u, f1_u))

def train_unary(train_data, val_data, conv_weight_decay = REGULARIZATION_STRENGTH):

  training_queue = ShuffleQueue(train_data)

  data_shape = [BATCH_SIZE, PATCH_SIZE, 1, N_FEATURES]
  labs_shape = [BATCH_SIZE, PATCH_SIZE, 1, 1]
  train_features = tf.placeholder(tf.float32, shape=data_shape)
  train_labels   = tf.placeholder(tf.int64,   shape=labs_shape)

  logits = unary(train_features,
                 is_training=True,
                 conv_weight_decay=conv_weight_decay,
                 dropout_keep_prob=DROPOUT_KEEP_PROB)
  l = loss(tf.reshape(logits, [-1, 2]), tf.reshape(train_labels, [-1]))
  train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(l)

  test_features = tf.placeholder(tf.float32)
  tf.get_variable_scope().reuse_variables()
  test_logits = unary(test_features, is_training=False)

  saver = tf.train.Saver(tf.get_collection(UNARY_VARIABLES))
  init_op = tf.global_variables_initializer()

  with tf.Session() as session:
    # Initialize
    session.run(init_op)

    def prediction(features, edge_features):
      features = features[np.newaxis, :, np.newaxis, :]
      logits = session.run(test_logits, feed_dict={test_features: features})
      return np.argmax(logits, axis=-1).flatten()

    BEST_VAL_SO_FAR = 0
    start = time.time()
    for step in range(TRAIN_STEPS+1):
      # Construct a bs-length numpy array
      features, _, labels, edge_labels = get_batch(training_queue)
      # Run a training step
      loss_val, _ = session.run(
        [l, train_op],
        feed_dict={train_features: features, train_labels: labels}
      )

      if step % 100 == 0:
        _,_,_,f1_validation = evaluate_unary(val_data, prediction)
        _,_,_,f1_train = evaluate_unary(train_data, prediction)
        if f1_validation > BEST_VAL_SO_FAR:
          best = True
          saver.save(session, os.path.join(CHECKPOINT_DIR, 'unary.ckpt'))
          BEST_VAL_SO_FAR = f1_validation
        else:
          best = False
        print("%10d: train=%.4f, val=%.4f %s" % (step, f1_train, f1_validation, '*' if best else ''))
    # saver.save(session, os.path.join(CHECKPOINT_DIR, 'unary.ckpt'))
  return f1_validation

def train_edge(train_data, val_data, conv_weight_decay = REGULARIZATION_STRENGTH):
  training_queue = ShuffleQueue(train_data)

  data_shape = [BATCH_SIZE, PATCH_SIZE-1, 1, N_EDGE_FEATURES]
  labs_shape = [BATCH_SIZE, PATCH_SIZE-1, 1, 1]
  train_features = tf.placeholder(tf.float32, shape=data_shape)
  train_labels   = tf.placeholder(tf.int64,   shape=labs_shape)

  logits = edge(train_features,
                is_training=True,
                conv_weight_decay=conv_weight_decay,
                dropout_keep_prob=DROPOUT_KEEP_PROB)
  l = loss(tf.reshape(logits, [-1, 4]), tf.reshape(train_labels, [-1]))
  train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(l)

  test_features = tf.placeholder(tf.float32)
  tf.get_variable_scope().reuse_variables()
  test_logits = edge(test_features, is_training=False)

  saver = tf.train.Saver(tf.get_collection(EDGE_VARIABLES))
  init_op = tf.global_variables_initializer()

  with tf.Session() as session:
    # Initialize
    session.run(init_op)

    def prediction(features):
      features = features[np.newaxis, :, np.newaxis, :]
      logits = session.run(test_logits, feed_dict={test_features: features})
      return np.argmax(logits, axis=-1).flatten()

    BEST_VAL_SO_FAR = 0
    for step in range(TRAIN_STEPS+1):
      # Construct a bs-length numpy array
      _, edge_features, labels, edge_labels = get_batch(training_queue)
      # Run a training step
      loss_val, _ = session.run(
        [l, train_op],
        feed_dict={train_features: edge_features, train_labels: edge_labels}
      )


      if step % 100 == 0:
        accuracy_validation = evaluate_edge(val_data, prediction)
        accuracy_train = evaluate_edge(train_data, prediction)
        if accuracy_validation > BEST_VAL_SO_FAR:
          best = True
          saver.save(session, os.path.join(CHECKPOINT_DIR, 'edge.ckpt'))
          BEST_VAL_SO_FAR = accuracy_validation
        else:
          best = False
        print("%10d: train=%.4f, val=%.4f %s" % (step, accuracy_train, accuracy_validation, '*' if best else ''))
    # saver.save(session, os.path.join(CHECKPOINT_DIR, 'edge.ckpt'))
  return accuracy_validation

def run_bootsrap(iterations, dataset, indices):
    structured_results = []
    unary_results = []

    for i in range(iterations):
      print("Iteration >>> {}".format(i))
      try:
        # get bootstrap datasets
        train_ind = resample(indices, n_samples=TRAIN_SIZE, replace=False)
        val_ind = resample(np.setdiff1d(indices,train_ind), n_samples=VAL_SIZE, replace=False)
        test_ind = np.setdiff1d(indices,np.append(train_ind, val_ind))

        train_data = dataset[train_ind]
        val_data = dataset[val_ind]
        test_data = dataset[test_ind]

        ## Network Training
        tf.reset_default_graph()
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_unary(train_data, val_data)
        tf.reset_default_graph()
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_edge(train_data, val_data)
        tf.reset_default_graph()

        # Evaluate Networks
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            str_results, una_results = test_structured(test_data)
        structured_results.append(str_results)
        unary_results.append(una_results)
        append_results([str_results], [una_results])
        with open("results/splits.txt", "a") as f:
          f.writelines(["split_{}\n".format(i), 
          ",".join(np.char.mod('%f', train_ind))+ "\n", 
          ",".join(np.char.mod('%f', val_ind))+ "\n", 
          ",".join(np.char.mod('%f', test_ind))+ "\n"])

      except Exception as e:
        print(e)

    return structured_results, unary_results

def append_results(structured_results, unary_results):
    structured_df = pd.DataFrame(structured_results, columns=["Accuracy", "Precision", "Recall", "F1"])
    structured_df.to_csv("results/Structured_Results.csv", index=False, mode="a", header=False)
    unary_df = pd.DataFrame(unary_results, columns=["Accuracy", "Precision", "Recall", "F1"])
    unary_df.to_csv("results/Unary_Results.csv", index=False, mode="a", header=False)

def main():
  if len(sys.argv) < 2:
    print("USAGE: python boostrap.py [iterations] [dataset]")
    sys.exit()
  iterations = int(sys.argv[1])

  if len(sys.argv) < 3:
    dataset = cleaneval
    indices = all_data_cleaneval
  elif sys.argv[2] == 'cleanEval':
    dataset = cleaneval
    indices = all_data_cleaneval
  elif sys.argv[2] == 'googleTrends':
    dataset = googletrends
    indices = all_data_googletrends
  else:
    dataset = cleaneval
    indices = all_data_cleaneval

  structured_results, unary_results = run_bootsrap(iterations, dataset, indices)


if __name__ == "__main__":
    main()
