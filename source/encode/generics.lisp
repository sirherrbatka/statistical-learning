(cl:in-package #:sl.encode)


(defgeneric size-required (encoder))

(defgeneric encode-column-value (encoder array bitmask offset row column))

(defgeneric make-encoder-for-type (type data-frame column))
