(cl:in-package #:statistical-learning.mp)


(defun make-supervised-model (parameters training-data target-data
                              &rest initargs &key &allow-other-keys)
  (make-model* parameters
               (apply #'make-training-state
                      parameters
                      :train-data (sl.data:wrap training-data)
                      :target-data (sl.data:wrap target-data)
                      initargs)))


(defun make-unsupervised-model (parameters data
                                &rest initargs &key &allow-other-keys)
  (make-model* parameters
               (apply #'make-training-state
                      parameters
                      :data (sl.data:wrap data)
                      initargs)))

