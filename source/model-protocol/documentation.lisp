(cl:in-package #:cl-grf.mp)


(docs:define-docs
  :formatter docs.ext:rich-aggregating-formatter

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
