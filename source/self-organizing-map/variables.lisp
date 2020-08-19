(cl:in-package #:sl.som)


(define-constant +linear-decay-final-alpha+ 0.05d0)
(define-constant +linear-decay-final-sigma+ 1.0d0)

(def <linear-decay> (make 'linear-decay))
(def <hill-decay> (make 'hill-decay))
