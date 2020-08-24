import numpy as np
import tensorflow as tf

class base_model():
    def __init__(self, task_type, model, loss_object, optimizer, eval_metric, batch_size,
                 slice_input, eval_metric_name):
        
        assert(task_type in ["classification", "regression"])
        
        self.task_type = task_type
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.loss_avg = tf.keras.metrics.Mean()
        self.eval_metric = eval_metric
        self.batch_size = batch_size
        
        self.slice_input = slice_input
        self.eval_metric_name = eval_metric_name
        
    def _loss(self, x, y):
        y_ = self.model(x)
        return self.loss_object(y_true=y, y_pred=y_)

    def _grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
    
    def fit(self, train_X, train_Y, val_X, val_Y, num_epochs):
        
        template = ("Epoch {:03d}: "
                "Train Loss: {:.3f}, " 
                "Train {eval_metric_name}: {:.3%}  "
                "Val Loss: {:.3f}, " 
                "Val {eval_metric_name}: {:.3%}  ")

        train_loss = np.zeros((num_epochs,), dtype=np.float32)
        train_eval_metric = np.zeros((num_epochs,), dtype=np.float32)
        val_loss = np.zeros((num_epochs,), dtype=np.float32)
        val_eval_metric = np.zeros((num_epochs,), dtype=np.float32)

        for epoch in range(num_epochs):
            
            train_loss[epoch], train_eval_metric[epoch] = self.evaluate(inputs=train_X,
                                                                        outputs=train_Y,
                                                                        is_training=True)
            
            val_loss[epoch], val_eval_metric[epoch] = self.evaluate(inputs=val_X,
                                                                    outputs=val_Y,
                                                                    is_training=False)
            print(template.format(epoch,
                                  train_loss[epoch],
                                  train_eval_metric[epoch],
                                  val_loss[epoch],
                                  val_eval_metric[epoch],
                                  eval_metric_name=self.eval_metric_name))


        return (train_loss, train_eval_metric, val_loss, val_eval_metric)
        
    
    def evaluate(self, inputs, outputs, is_training):
        
        self.loss_avg.reset_states()
        self.eval_metric.reset_states()
        
        num_samples = outputs.shape[0]
        
        # using batches of <self.batch_size>
        for i in range(0, num_samples, self.batch_size):

                x = self.slice_input(inputs, i, i + self.batch_size)
                y = outputs[i : i + self.batch_size, ...]
                
                if is_training:
                    loss_value, grads = self._grad(x, y)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                else:
                    loss_value = self._loss(x, y)

                self.loss_avg.update_state(loss_value)  
                
                if self.task_type is "regression":
                    self.eval_metric.update_state(tf.keras.backend.flatten(y), 
                                                  tf.keras.backend.flatten(self.model(x)))
                elif self.task_type is "classification":
                    self.eval_metric.update_state(y,
                                                  self.model(x))
        
        return (self.loss_avg.result(), self.eval_metric.result())