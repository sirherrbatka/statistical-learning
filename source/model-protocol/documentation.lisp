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
     "Will signal type-error if either target-data or train-data are not cl-grf.data:data-matrix.")
    :returns "A new model, class depending on the PARAMETERS."
    :notes ("Methods dispatches on the PARAMETERS."
            "TRAIN-DATA and TARGET-DATA validated in the around method.")))

  (function
   predict
   (:description "Use model do predict values."
    :returns "Vector holding predictions. Specific type of predictions depends on the model."
    :exceptional-situations "Will signal type-error if data is not of the type cl-grf.data:data-matrix."
    :notes "Argument validation is performed in the around method."))

  (function
   cross-validation
   (:description "Perform cross validation of the model with data."
    :returns "Average performance metric."))

  (type
   confusion-matrix
   (:description "Holds data required for calculating precision, recall, specifity and accuracy. Is used as a primary performance metric for classification tasks."
    :see-also (precsion recall specificy accuracy true false total negative positive true-positive true-negative false-positive false-negative positive negative)))

  (function
   total
   (:description "Get a complete number of cases out of the CONFUSION-MATRIX."))

  (function
   true-positive
   (:description "Get a complete number of true positive cases out of the CONFUSION-MATRIX."))

  (function
   false-positive
   (:description "Get a complete number of false positive cases out of the CONFUSION-MATRIX."))

  (function
   true-negative
   (:description "Get a complete number of true negative cases out of the CONFUSION-MATRIX."))

  (function
   false-negative
   (:description "Get a complete number of false negative cases out of the CONFUSION-MATRIX."))

  (function
   true
   (:description "Get a complete number of true cases (both false-negative and true positive) out of the CONFUSION-MATRIX."))

  (function
   f1-score
   (:description "Calculates F1-SCORE from the CONFUSION-MATRIX."
    :returns "DOUBLE-FLOAT representing F1-SCORE."))

  (function
   precision
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
