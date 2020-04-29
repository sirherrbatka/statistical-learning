(cl:in-package #:cl-grf.data)


(defmacro bind-data-matrix-dimensions ((data-points-count
                                        attributes-count
                                        data-matrix)
                                       &body body)
  `(bind (((,data-points-count ,attributes-count)
           (data-matrix-dimensions ,data-matrix)))
     (declare (type fixnum ,data-points-count ,attributes-count))
     ,@body))
