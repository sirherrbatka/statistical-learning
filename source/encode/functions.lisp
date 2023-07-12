(cl:in-package #:sl.encode)


(defun make-encoders (data-frame)
  (iterate
    (with header = (vellum.table:header data-frame))
    (for i from 0 below (vellum.header:column-count header))
    (for type = (vellum.header:column-type header i))
    (collecting (make-encoder-for-type type data-frame i))))


(defun encode (encoders data-frame)
  (assert encoders)
  (bind ((offsets (serapeum:scan #'+ encoders :key #'size-required :initial-value 0))
         (number-of-columns (reduce #'+ encoders :key #'size-required))
         (array (make-array (list (vellum:row-count data-frame) number-of-columns)
                            :element-type 'double-float))
         (bitmask (make-array (list (vellum:row-count data-frame) number-of-columns)
                              :element-type 'bit)))
    (vellum:transform data-frame
                      (vellum:bind-row ()
                        (iterate
                          (for column from 0)
                          (for encoder in encoders)
                          (for offset in offsets)
                          (encode-column-value encoder array bitmask offset vellum.table:*current-row* column))))
    (sl.data:wrap array bitmask)))
