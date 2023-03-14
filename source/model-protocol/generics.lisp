(cl:in-package #:statistical-learning.model-protocol)


(sl.common:defgeneric/proxy make-model* ((parameters)
                                         training-state))

(sl.common:defgeneric/proxy make-training-state
    ((parameters)
     &rest initargs
     &key &allow-other-keys))

(sl.common:defgeneric/proxy sample-training-state*
    ((parameters)
     state
     &key
     train-attributes
     target-attributes
     initargs
     &allow-other-keys))

(sl.common:defgeneric/proxy
    sample-training-state-info
    ((parameters)
     state
     &key
     train-attributes
     target-attributes
     &allow-other-keys)
  (:method-combination append :most-specific-last))

(defgeneric predict (model data &optional parallel))
(defgeneric parameters (model))
(defgeneric training-parameters (state))
(defgeneric (setf training-parameters) (new-value state))
(defgeneric target-data (state))
(defgeneric train-data (state))
(defgeneric weights-data (state))
(defgeneric cache (state key))
(defgeneric model-cache (state key))
(defgeneric (setf cache) (new-value state key))
(defgeneric (setf model-cache) (new-value state key))
