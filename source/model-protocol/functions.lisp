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


(defun make-model (parameters training-data target-data
                   &rest initargs &key weights &allow-other-keys)
  (make-model* parameters
               (apply #'make-training-state
                      parameters
                      training-data
                      target-data
                      initargs)))
