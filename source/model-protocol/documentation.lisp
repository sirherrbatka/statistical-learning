(cl:in-package #:cl-grf.mp)


(docs:define-docs
  :formatter docs.ext:rich-aggregating-formatter

  (function
   make-model
   (:description "Construct trained model given data and parameters"
    :exceptional-situations
    ("Zero attributes in the train-data or target-data will signal invalid-argument-value."
     "Zero data-points in the train-data or target-data will signal invalid-argument-value."
     "Unequal number of data-points in train-data and target-data will signal incompatible-arguments."
     "Will signal type-error if either target-data or train-data are not cl-grf.data:data-matrix."
     )
    :notes ("Methods dispatches on the PARAMETERS."
            "TRAIN-DATA and TARGET-DATA validated in the around method.")))

  (function
   cross-validation
   (:description "Perform cross validation of the model with data."
    :returns "Average performance metric."))

  (type
   confusion-matrix
   (:description "Holds data required for calculating precision, recall, specifity and accuracy."
    :see-also (precsion recall specificy accuracy true false total negative positive true-positive true-negative false-positive false-negative total-positive total-negative)))

  (function
   precsion
   (:description "Calculates precsion from the confusion-matrix."
    :returns "DOUBLE-FLOAT representing precsion value."))

  (function
   recall
   (:description "Calculates recall from the confusion-matrix."
    :returns "DOBULE-FLOAT representing recall value."))

  (function
   specificity
   (:description "Calculates specificity from the confusion-matrix."
    :returns "DOBULE-FLOAT representing specificity value."))

  (function
   accuracy
   (:description "Calculates accuracy from the confusion-matrix."
    :returns "DOBULE-FLOAT representing accuracy value."))
  )
