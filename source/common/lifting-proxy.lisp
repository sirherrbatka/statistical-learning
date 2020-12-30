(cl:in-package #:statistical-learning.common)


(defgeneric next-proxy (proxy))


(defgeneric proxy (proxy-enabled))


(defmethod next-proxy ((proxy (eql nil)))
  nil)


(defmethod proxy ((proxy-enabled (eql nil)))
  nil)


(defclass lifting-proxy ()
  ((%next-proxy :initarg :next-proxy
                :reader next-proxy))
  (:default-initargs :next-proxy nil))


(defclass proxy-enabled ()
  ((%proxy :initarg :proxy
           :accessor proxy))
  (:default-initargs :proxy nil))


(defun lift (object proxy-class &rest args)
  (setf (proxy object) (apply #'make proxy-class
                              :next-proxy (proxy object)
                              args))
  object)


(defmethod cl-ds.utils:cloning-information
    append ((object proxy-enabled))
  '((:proxy proxy)))


(defun strip (object &optional (proxy (next-proxy (proxy object))))
  (cl-ds.utils:quasi-clone* object
    :proxy proxy))


(defmacro defgeneric/proxy (name arguments
                            &optional options (strip t))
  (bind (((:flet /proxy (symbol))
          (intern (format nil "~a/PROXY" symbol)))
         (function-arguments (mapcar (lambda (x)
                                       (if (listp x)
                                           (first x)
                                           x))
                                     arguments))
         ((:values required optional rest key)
          (parse-ordinary-lambda-list function-arguments))
         (setfp (listp name))
         (with-proxy-name (if setfp
                              `(setf ,(/proxy (second name)))
                              (/proxy name)))
         (real-arguments (append required
                                 (mapcar #'first optional)
                                 (apply #'append (mapcar #'car key))
                                 (when rest
                                   (list rest))))
         (proxy-objects (~>> arguments
                             (remove-if #'atom)
                             (mapcar #'first)))
         (proxy-arguments (mapcar #'/proxy proxy-objects))
         ((:flet generic-lambda-list
            (&optional (proxy-arguments proxy-arguments)))
          (if setfp
              `(,(first function-arguments)
                ,@proxy-arguments
                ,@(rest function-arguments))
              `(,@proxy-arguments
                ,@function-arguments))))
    `(progn
       (defgeneric ,with-proxy-name ,(generic-lambda-list)
         ,@(if (null options)
               nil
               (list options)))
       ,@(when strip
           (iterate
             (for arg in proxy-arguments)
             (for list = (substitute (list arg 'lifting-proxy)
                                     arg
                                     proxy-arguments))
             (for next = (if strip
                             (substitute `(next-proxy ,arg)
                                         arg
                                         proxy-arguments)))
             (collect `(defmethod ,with-proxy-name
                           ,@(remove-if #'keywordp options)
                         ,(generic-lambda-list list)
                         ,@(when rest
                             `((declare (ignore ,@(mapcar #'cadar key)))))
                         ,(if setfp
                              `(,(if rest 'apply 'funcall)
                                #',with-proxy-name
                                ,(first real-arguments)
                                ,@next
                                ,@(rest real-arguments))
                              `(,(if rest 'apply 'funcall)
                                #',with-proxy-name
                                ,@next
                                ,@real-arguments))))))
       (defun ,name (,@function-arguments)
         ,@(when rest
             `((declare (ignore ,@(mapcar #'cadar key)))))
         ,(if setfp
             `(,(if rest 'apply 'funcall)
               #',with-proxy-name
               ,(first real-arguments)
               ,@(iterate
                   (for p in proxy-objects)
                   (collect `(proxy ,p)))
              ,@(rest real-arguments))
             `(,(if rest 'apply 'funcall)
               #',with-proxy-name
               ,@(iterate
                   (for p in proxy-objects)
                   (collect `(proxy ,p)))
               ,@real-arguments))))))
