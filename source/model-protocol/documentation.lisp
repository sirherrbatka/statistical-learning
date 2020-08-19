(cl:in-package #:statistical-learning.mp)


(docs:define-docs
  :formatter docs.ext:rich-aggregating-formatter

  (function
   make-model
   (:description "Construct trained model given data and parameters"
    :exceptional-situations
    ("Zero attributes in the train-data or target-data will signal invalid-argument-value."
     "Zero data-points in the train-data or target-data will signal invalid-argument-value."
     "Unequal number of data-points in train-data and target-data will signal incompatible-arguments."
     "Will signal type-error if either target-data or train-data are not statistical-learning.data:data-matrix.")
    :returns "A new model, class depending on the PARAMETERS."
    :notes ("Methods dispatches on the PARAMETERS."
            "TRAIN-DATA and TARGET-DATA validated in the after method.")))

  (function
   predict
   (:description "Use model do predict values."
    :returns "Vector holding predictions. Specific type of predictions depends on the model."
    :exceptional-situations "Will signal type-error if data is not of the type statistical-learning.data:data-matrix."
    :notes "Argument validation is performed in the after method.")))
