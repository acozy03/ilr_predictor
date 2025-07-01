import tensorflow as tf

class MacroPrecision(tf.keras.metrics.Metric):
    


    def __init__(self, num_classes, name='macro_precision', **kwargs):
        #initialize the metric with the number of classes and name
        #add two weights for true positives and false positives
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(shape=(num_classes,), initializer='zeros', name='tp')
        self.false_positives = self.add_weight(shape=(num_classes,), initializer='zeros', name='fp')

    def update_state(self, y_true, y_pred, sample_weight=None):
        #update the state of the metric with the true and predicted labels
        #convert the predicted labels to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        tp_updates = tf.zeros(shape=(self.num_classes,), dtype=self.dtype)
        fp_updates = tf.zeros(shape=(self.num_classes,), dtype=self.dtype)

        for class_id in range(self.num_classes):
            #calculate true positives and false positives for each class
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), self.dtype))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(tf.equal(y_true, class_id)), tf.equal(y_pred, class_id)), self.dtype))

            tp_updates = tf.tensor_scatter_nd_add(tp_updates, [[class_id]], [tp])
            fp_updates = tf.tensor_scatter_nd_add(fp_updates, [[class_id]], [fp])
            
        #assign/add the calculated values to the respective weights
        self.true_positives.assign_add(tp_updates)
        self.false_positives.assign_add(fp_updates)

    def result(self):
        #calculate precision for each class and return the mean precision
        #add a small value to avoid division by zero
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        return tf.reduce_mean(precision)

    #reset the states of the metric
    #this is called at the beginning of each epoch or when the metric is reset
    def reset_states(self):
        for var in self.variables:
            var.assign(tf.zeros_like(var))
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MacroRecall(tf.keras.metrics.Metric):
    

    def __init__(self, num_classes, name='macro_recall', **kwargs):
        #initialize the metric with the number of classes and name
        #add two weights for true positives and false negatives
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(shape=(num_classes,), initializer='zeros', name='tp')
        self.false_negatives = self.add_weight(shape=(num_classes,), initializer='zeros', name='fn')

    def update_state(self, y_true, y_pred, sample_weight=None):
        #update the state of the metric with the true and predicted labels
        #convert the predicted labels to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        tp_updates = tf.zeros(shape=(self.num_classes,), dtype=self.dtype)
        fn_updates = tf.zeros(shape=(self.num_classes,), dtype=self.dtype)
    
        for class_id in range(self.num_classes):
            #calculate true positives and false negatives for each class
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), self.dtype))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.logical_not(tf.equal(y_pred, class_id))), self.dtype))

            tp_updates = tf.tensor_scatter_nd_add(tp_updates, [[class_id]], [tp])
            fn_updates = tf.tensor_scatter_nd_add(fn_updates, [[class_id]], [fn])
            
        #assign/add the calculated values to the respective weights
        self.true_positives.assign_add(tp_updates)
        self.false_negatives.assign_add(fn_updates)

    def result(self):
        #calculate recall for each class and return the mean recall
        #add a small value to avoid division by zero
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        return tf.reduce_mean(recall)

    #reset the states of the metric
    #this is called at the beginning of each epoch or when the metric is reset
    def reset_states(self):
        for var in self.variables:
            var.assign(tf.zeros_like(var))
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MacroF1Score(tf.keras.metrics.Metric):
    

    def __init__(self, num_classes, name='macro_f1', **kwargs):
        #initialize the metric with the number of classes and name
        #add two weights for precision and recall metrics
        super().__init__(name=name, **kwargs)
        self.precision_metric = MacroPrecision(num_classes)
        self.recall_metric = MacroRecall(num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        #update the state of the metric with the true and predicted labels using the precision and recall metrics
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision_metric.result()
        r = self.recall_metric.result()
        #calculate F1 score using the precision and recall values
        #add a small value to avoid division by zero
        return 2 * ((p * r) / (p + r + 1e-7))

    #reset the states of the metric
    #this is called at the beginning of each epoch or when the metric is reset
    def reset_states(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.precision_metric.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)