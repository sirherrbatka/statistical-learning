(cl:in-package #:sl.encode)


(defmethod size-required ((encoder identity-enocder))
  1)


(defmethod encode-column-value ((encoder identity-enocder) array bitmask offset row column)
  (let ((value (vellum:rr column)))
    (if (eq value :null)
        (setf (aref bitmask row offset) 0)
        (setf (aref array row offset) (coerce value 'single-float)
              (aref bitmask row offset) 1))))


(defmethod size-required ((encoder one-hot-encoder))
  (~> encoder content hash-table-count))


(defmethod size-required ((encoder boolean-encoder))
  1)


(defmethod encode-column-value ((encoder one-hot-encoder) array bitmask offset row column)
  (let ((value (vellum:rr column)))
    (if (eq value :null)
        (iterate
          (for i from offset)
          (repeat (size-required encoder))
          (setf (aref bitmask row i) 0))
        (iterate
          (for i from offset)
          (repeat (size-required encoder))
          (setf (aref bitmask row i) 1
                (aref array row i) 0.0)
          (finally
           (setf (aref array row (+ offset (gethash value (content encoder))))
                 1.0))))))


(defmethod make-encoder-for-type ((type (eql 'string)) data-frame column)
  (make 'one-hot-encoder
        :content (vellum:pipeline (data-frame)
                   (cl-ds.alg:on-each (lambda (&optional ignored)
                                        (declare (ignore ignored))
                                        (vellum:rr column)))
                   (cl-ds.alg:without (curry #'eql :null))
                   (cl-ds.alg:enumerate :test 'equal))))


(defmethod make-encoder-for-type ((type (eql 'single-float)) data-frame column)
  (make 'identity-enocder))


(defmethod make-encoder-for-type ((type (eql 'fixnum)) data-frame column)
  (make 'identity-enocder))


(defmethod make-encoder-for-type ((type (eql 'boolean)) data-frame column)
  (make 'boolean-encoder))


(defmethod encode-column-value ((encoder boolean-encoder) array bitmask offset row column)
  (let ((value (vellum:rr column)))
    (if (eq value :null)
        (setf (aref bitmask row offset) 0)
        (setf (aref array row offset) (if value 1.0 0.)
              (aref bitmask row offset) 1))))
