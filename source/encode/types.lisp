(cl:in-package #:sl.encode)


(defclass fundamental-encoder ()
  ())


(defclass one-hot-encoder (fundamental-encoder)
  ((%content :initarg :content
             :reader content)))


(defclass identity-enocder (fundamental-encoder)
  ())


(defclass boolean-encoder (fundamental-encoder)
  ())
