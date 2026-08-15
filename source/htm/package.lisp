(cl:defpackage #:statistical-learning.hierachical-temporal-memory
  (:use #:common-lisp #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.htm)
  (:local-nicknames
   )
  (:export
   ;; Main HTM class
   #:htm-parameters

   ;; HTM parameter accessors
   #:htm-neurons-per-column
   #:htm-active-neurons-per-column
   #:htm-min-activation-threshold
   #:htm-segments-per-neuron
   #:htm-min-segment-threshold
   #:htm-permanence-increment
   #:htm-permanence-decrement
   #:htm-permanence-threshold
   #:htm-learning-rate
   #:htm-decay-rate
   #:htm-decay-threshold

   ;; HTM state accessors
   #:neuron-permanence
   #:segment-permanence
   #:active-neurons
   #:active-segments
   #:prediction-error))
