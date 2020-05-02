(cl:in-package #:cl-grf.data)


(defmacro bind-data-matrix-dimensions (bindings &body body)
  (let ((variables (flatten (mapcar (curry #'take 2) bindings))))
    `(bind ,(mapcar (lambda (binding)
                      (list (take 2 binding)
                            `(data-matrix-dimensions ,(third binding))))
                    bindings)
       (declare (type fixnum ,@variables)
                (ignorable ,@variables))
       ,@body)))
