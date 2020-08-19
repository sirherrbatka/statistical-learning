(cl:in-package #:statistical-learning.mp)


(defun sample-training-state (state &key data-points
                                      train-attributes
                                      target-attributes
                                      initargs)
  (sample-training-state* (training-parameters state)
                          state
                          :data-points data-points
                          :train-attributes train-attributes
                          :target-attributes target-attributes
                          :initarg initargs))


(defun make-supervised-model (parameters training-data target-data
                              &rest initargs &key &allow-other-keys)
  (make-model* parameters
               (apply #'make-training-state
                      parameters
                      :train-data training-data
                      :target-data target-data
                      initargs)))


(defun make-unsupervised-model (parameters data
                                &rest initargs &key &allow-other-keys)
  (make-model* parameters
               (apply #'make-training-state
                      parameters
                      :data data
                      initargs)))

