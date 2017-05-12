import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar
import gensim.models
from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon
"""

wordDict = {}
posDict = {}
labelDict = {}
system = None

def genDictionaries(sents, trees):

    """
    Generate Dictionaries for word, pos, and arc_label
    Since we will use same embedding array for all three groups,
    each element will have unique ID
    """
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n+1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]

def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]

def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]

"""
This method will extract the features of given configuration
these features include total 48 Tokens which contains 18 words
18 POS tags and 12 labels.
"""
def getFeatures(c):
    word = []
    pos = []
    label = []

    """
    Here we are extracting top 3 words and their part of speech tags as features
    """
    for i in [0,1,2]:
        index = c.getStack(i)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))

    """
    Here we are extracting top 3 buffer words and their part of speech tags as features
    """
    for i in [0,1,2]:
        index = c.getBuffer(i)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))

    """
    Here we are extracting words, POS tags and labels of stack's top words's left child
    right child, second left child, second right child, left child of left child and right child of
    right child, At the end we are appending all 3 kinds of features and return
    """

    for i in [0,1]:
        k = c.getStack(i)
        index = c.getLeftChild(k,1)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(k,1)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getLeftChild(k,2)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(k,2)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getLeftChild(c.getLeftChild(k,1),1)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(c.getRightChild(k,1),1)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

    features = []
    features.extend(word)
    features.extend(pos)
    features.extend(label)
    return features


    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

def genTrainExamples(sents, trees):

    """
    Generate train examples
    Each configuration of dependency parsing will give us one training instance
    Each instance will contains:
        WordID, PosID, LabelID as described in the paper(Total 48 IDs)
        Label for each arc label:
            correct ones as 1,
            appliable ones as 0,
            non-appliable ones as -1
    """
    numTrans = system.numTransitions()
    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = system.initialConfiguration(sents[i])
            while not system.isTerminal(c):
                oracle = system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = system.transitions[j]
                    if t == oracle: label.append(1.)
                    elif system.canApply(c, t): label.append(0.)
                    else: label.append(-1.)

                features.append(feat)
                labels.append(label)
                c = system.apply(c, oracle)
    return features, labels

def forward_pass(train_inputs_embed, w1, w2, biases):
    
    hidden_layer = tf.pow(tf.add(tf.matmul(train_inputs_embed, w1),biases),3)
    logits = tf.matmul(hidden_layer, w2)
    return logits

    """
    =======================================================

    Implement the forwrad pass described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =======================================================
    """

if __name__ == '__main__':

    # Load all dataset
    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    # Load pre-trained word embeddings
    dictionary, word_embeds = pickle.load(open('word2vec.model', 'rb'))


    # Create embedding array for word + pos + arc_label
    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = list(wordDict.keys())
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary : index = dictionary[w]
            elif w.lower() in dictionary : index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size)*0.02-0.01
    print("Found embeddings: ", foundEmbed, "/", len(knownWords))

    # Get a new instance of ParsingSystem with arc_labels
    system = ParsingSystem(list(labelDict.keys()))

    print("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print("Done.")



    graph = tf.Graph()

    with graph.as_default():
        embeddings = tf.Variable(embedding_array, dtype=tf.float32)

        """
        ===================================================================

        Define the computational graph with necessary variables.
        You may need placeholders of:
            train_inputs
            train_labels
            test_inputs

        Implement the loss function described in the paper

        ===================================================================
        """

        test_inputs = tf.placeholder(tf.int32)

        train_labels = tf.placeholder(tf.float32, shape=[Config.batch_size, len(trainLabels[0])])
        train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])

        test_inputs_embed = tf.nn.embedding_lookup(embeddings, test_inputs)
        test_inputs_embed = tf.reshape(test_inputs_embed, [1, Config.n_Tokens*Config.embedding_size])

        train_inputs_embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        train_inputs_embed = tf.reshape(train_inputs_embed,[Config.batch_size, Config.n_Tokens*Config.embedding_size])


        w1 = tf.Variable(tf.random_normal([Config.n_Tokens*Config.embedding_size, Config.hidden_size],stddev=1.0 / math.sqrt(Config.hidden_size)))
        w2 = tf.Variable(tf.random_normal([Config.hidden_size, len(trainLabels[0])], stddev=1.0 / math.sqrt(len(trainLabels[0]))))

        biases = tf.Variable(tf.zeros([Config.hidden_size]))
        logits = forward_pass(train_inputs_embed, w1, w2, biases)

        temp_lab = tf.arg_max(train_labels, dimension=1)

        r = 0.5 * Config.lam * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(biases) + tf.nn.l2_loss(embeddings))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=temp_lab, logits=logits)) + r
        test_pred = tf.nn.softmax(forward_pass(test_inputs_embed,w1, w2, biases))
        optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)

        # Compute Gradients
        grads = optimizer.compute_gradients(loss)

        # Gradient Clipping
        clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        app = optimizer.apply_gradients(clipped_grads)
        init = tf.global_variables_initializer()

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:
        init.run()
        print("Initailized")

        average_loss = 0
        for step in range(num_steps):
            start = (step*Config.batch_size)%len(trainFeats)
            end = ((step+1)*Config.batch_size)%len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = sess.run([app, loss], feed_dict=feed_dict)
            average_loss += loss_val

            # Display average loss
            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print ("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            
            # Print out the performance on dev set
            if step % Config.validation_step == 0 and step != 0:
                print ("\nTesting on dev set at step ", step)
                predTrees = []
                for sent in devSents:
                    numTrans = system.numTransitions()
                    c = system.initialConfiguration(sent)
                    while not system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(test_pred, feed_dict={test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and system.canApply(c, system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = system.transitions[j]

                        c = system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = system.evaluate(devSents, predTrees, devTrees)
                print(result)
        print("Optimization Finished.")

        print("Start predicting on test set")
        predTrees = []
        for sent in testSents:
            numTrans = system.numTransitions()
        
            c = system.initialConfiguration(sent)
            while not system.isTerminal(c):
                feat = getFeatures(c)
                pred = sess.run(test_pred, feed_dict={test_inputs: feat})
                optScore = -float('inf')
                optTrans = ""
        
                for j in range(numTrans):
                    if pred[0, j] > optScore and system.canApply(c, system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = system.transitions[j]
        
                c = system.apply(c, optTrans)
        
            predTrees.append(c.tree)
        print("Store the test results.")
        Util.writeConll('result.conll', testSents, predTrees)

