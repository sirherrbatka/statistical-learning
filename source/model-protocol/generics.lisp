(cl:in-package #:statistical-learning.model-protocol)


(defgeneric make-model* (parameters training-state))
(defgeneric predict (model data &optional parallel))
(defgeneric parameters (model))
(defgeneric make-training-state (parameters &rest initargs &key &allow-other-keys))
(defgeneric sample-training-state* (parameters state
                                    &key
                                      data-points
                                      train-attributes
                                      target-attributes
                                      initargs
                                    &allow-other-keys))
(defgeneric sample-training-state-info (parameters state
                                        &key data-points
                                          train-attributes
                                          target-attributes
                                        &allow-other-keys)
  (:method-combination append :most-specific-last))

(defgeneric training-parameters (state))
(defgeneric (setf training-parameters) (new-value state))
(defgeneric target-data (state))
(defgeneric train-data (state))
(defgeneric weights-data (state))
