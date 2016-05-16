;(function (window, document) {
  "use strict";

  /*
   * FaceDetect
   * A class for detecting faces
   * @param DOMObject webcam
   * @param DOMObject canvas
   */
  var FaceDetect = function(webcam, canvas) {
    var self = this;

    if (!self.hasGetUserMedia()) {
      throw("getUserMedia() is not supported in your browser");
    }

    self.webcam = webcam;
    self.canvas = canvas;

    self.props = [];
    self.propTypes = {};

    self.maxWorkSize = 160;
    //Change detected face if it changes by a certain amount
    self.rectChangeThresh = 4;

    // Range of rotation of faces that should be detected
    // In radians
    self.rotationRange = 1;
    self.rotationStep = 0.5;

    self.debug = true;

    //Init fallback functions
    self.polyfills();

    self.initCamera();
  }

  /**
   * Add new prop to list
   * @param String url
   * @param String type
   */
  FaceDetect.prototype.addProp = function(url, type) {
    var self = this;

    if (!(type in self.propTypes)) throw("Prop must be of an allowed type");

    var image = new Image();
    image.src = url;
    //Only add the image if it loads
    image.addEventListener("load", function() {
      self.props.push({
        "image": image,
        "type": type
      })
    })

    return true;
  }

  /**
   * Add new prop type
   * @param String name
   * @param Float originX
   * @param Float originY
   * @param String ratio (width, height)
   */
  FaceDetect.prototype.addPropType = function(name, originX, originY, ratio) {
    var self = this;

    if (originX < 0 || originX > 1 || originY < 0 || originY > 1) {
      throw("Origin must be a float between 0 and 1")
    }

    if (ratio !== "width" && ratio !== "height") throw("Ratio must be based upon width or height");

    self.propTypes[name] = {
      "originX": originX,
      "originY": originY,
      "ratio": ratio
    };

    return true;
  }

  FaceDetect.prototype.drawProps = function(rectX, rectY, rectWidth, rectHeight) {
    var self = this;

    var prop, i, originX, originY, type;

    for (i = 0; i < self.props.length; i++) {
      prop = self.props[i];

      self.ctx.drawImage(prop["image"], rectX, rectY, rectWidth, rectHeight);
    }
  }

  FaceDetect.prototype.polyfills = function() {
    var self = this;

    if (!("FaceDetectUserMedia" in navigator)) navigator.FaceDetectUserMedia = (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
    if (!("requestAnimFrame" in window)) {
      window.requestAnimFrame = (function(){
        return  window.requestAnimationFrame       ||
                window.webkitRequestAnimationFrame ||
                window.mozRequestAnimationFrame    ||
                function( callback ){
                  window.setTimeout(callback, 1000 / 60);
                };
      })();
    }
  }

  FaceDetect.prototype.hasGetUserMedia = function() {
    var self = this;
    return !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
  };

  FaceDetect.prototype.initCamera = function() {
    var self = this;

    var attempts = 0;
    var findVideoSize = function() {
      if(self.webcam.videoWidth > 0 && self.webcam.videoHeight > 0) {
        self.cameraHeight = self.webcam.videoHeight;
        self.cameraWidth = self.webcam.videoWidth;
        self.ready();
      } else {
          if(attempts < 10) {
            attempts++;
            window.setTimeout(findVideoSize.bind(self), 200);
          } else {
            self.cameraHeight = 480;
            self.cameraWidth = 640;
            self.ready();
          }
      }
    };

    navigator.FaceDetectUserMedia({video: true, audio: false}, function(localMediaStream) {
      self.webcam.src = window.URL.createObjectURL(localMediaStream);

      self.webcam.addEventListener("loadeddata", function() {
        findVideoSize();
      })
    }, function() {
      throw("Couldn't load stream")
    });
  };

  FaceDetect.prototype.ready = function() {
    var self = this;

    self.ctx = self.canvas.getContext('2d');

    self.ctx.fillStyle = "rgb(0,255,0)";
    self.ctx.strokeStyle = "rgb(0,255,0)";

    // Calculate scale for work canvas
    var scale = Math.min(self.maxWorkSize/self.cameraWidth, self.maxWorkSize/self.cameraHeight);
    var scaledWidth = (self.canvas.width * scale) | 0;
    var scaledHeight = (self.canvas.height * scale) | 0;

    self.img_u8 = new jsfeat.matrix_t(scaledWidth, scaledHeight, jsfeat.U8_t | jsfeat.C1_t);
    self.workCanvas = document.createElement('canvas');
    self.workCanvas.width = scaledWidth;
    self.workCanvas.height = scaledHeight;
    self.work_ctx = self.workCanvas.getContext('2d');

    jsfeat.bbf.prepare_cascade(jsfeat.bbf.face_cascade);

    //Start tick function
    window.requestAnimFrame(self.tick.bind(self));
  };

  FaceDetect.prototype.tick = function() {
    var self = this;

    window.requestAnimFrame(self.tick.bind(self));

    if (self.webcam.readyState === self.webcam.HAVE_ENOUGH_DATA) {
      //Draw newest input from webcam to canvas
      self.ctx.drawImage(self.webcam, 0, 0, self.canvas.width, self.canvas.height);

      var rects = [];
      var originX = self.workCanvas.width / 2;
      var originY = self.workCanvas.height / 2;

      var halfAngleRange = self.rotationRange / 2;

      var angle, tempRects, imageData, pyr, currentRect, r, o; 

      for (angle = -halfAngleRange; angle <= halfAngleRange; angle += self.rotationStep) {

        //Translate and rotate current frame based upon current angle (in radians)
        self.work_ctx.translate(originX, originY);
        self.work_ctx.rotate(angle);
        self.work_ctx.drawImage(self.webcam, -self.workCanvas.width / 2, -self.workCanvas.height / 2, self.workCanvas.width, self.workCanvas.height);
        self.work_ctx.rotate(-angle)
        self.work_ctx.translate(-originX, -originY);
    
        // Grab image data from rotated canvas
        imageData = self.work_ctx.getImageData(0, 0, self.workCanvas.width, self.workCanvas.height);

        // Run face classifiers on captured image data
        jsfeat.imgproc.grayscale(imageData.data, self.workCanvas.width, self.workCanvas.height, self.img_u8);
        pyr = jsfeat.bbf.build_pyramid(self.img_u8, 24*2, 24*2, 4);
        tempRects = jsfeat.bbf.group_rectangles(jsfeat.bbf.detect(pyr, jsfeat.bbf.face_cascade), 1);
        
        //Translate found faces to coordinates of image
        for (currentRect = 0; currentRect < tempRects.length; currentRect++){
            tempRects[currentRect].x = tempRects[currentRect].x - originX;
            tempRects[currentRect].y = tempRects[currentRect].y - originY;

            //Calculate radius of rotated boxes
            r = Math.sqrt(Math.pow(tempRects[currentRect].x, 2) + Math.pow(tempRects[currentRect].y, 2));
            //Calculate original origin
            o = Math.atan(tempRects[currentRect].y / tempRects[currentRect].x);

            if (tempRects[currentRect].x < 0){
              o = o - Math.PI;
            }

            //Translate x and y coords to non rotated image
            tempRects[currentRect].x = (r * Math.cos(o - angle));
            tempRects[currentRect].y = (r * Math.sin(o - angle));

            tempRects[currentRect]["angle"] = angle;
        }
        rects = rects.concat(tempRects);
      }

      // draw only most confident face
      self.drawFaces(rects, self.canvas.width / self.img_u8.cols, 1);
    }
  };

  FaceDetect.prototype.drawFaces = function(rects, sc, max) {
    var self = this;

    //Sort rects by their confidence value
    var on = rects.length;
    if(on && max) {
        jsfeat.math.qsort(rects, 0, on-1, function(a, b) { return (b.confidence < a.confidence); })
    }
    var n = max || on;
    n = Math.min(n, on);

    var r, xChange, yChange, rectX, rectY, rectWidth, rectHeight;
    var originX = self.canvas.width / 2;
    var originY = self.canvas.height / 2;

    for(var i = 0; i < n; ++i) {
      r = rects[i];

      //Set the current rectangle if isn't set yet
      if (typeof(self.currRect) === "undefined") self.currRect = r;

      //Calculate if rectangle has changed significantly from previous one
      var rectDiff = Math.abs(self.currRect.x - r.x) + Math.abs(self.currRect.x - r.x) + Math.abs(self.currRect.width - r.width) + Math.abs(self.currRect.height - r.height);
      if (rectDiff < self.rectChangeThresh) {
        r = self.currRect;
      } else {
        self.currRect = r;
      }

      //Change origin of canvas to center of image and rotate based on rotation of box
      self.ctx.translate(originX, originY);
      self.ctx.rotate(-r.angle);

      //Adjust for slight offsets based upon the angle
      yChange = 0;
      xChange = 0;
      if (r.angle < 0) {
        xChange = -1 * (r.width / 2);
      } else if (r.angle > 0) {
        yChange = -1 * (r.height / 2);
        xChange = (r.width / 4);
      }

      //Calculate coordinates for face rectangle
      rectX = ((r.x + xChange) * sc) | 0;
      rectY = ((r.y + yChange) * sc) | 0;
      rectWidth = (r.width * sc) | 0;
      rectHeight = (r.height * sc) | 0

      //Draw the props
      self.drawProps(rectX, rectY, rectWidth, rectHeight);

      //Draw face rect on canvas
      if (self.debug) self.ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);

      //Translate canvas back to match the image
      self.ctx.rotate(r.angle);
      self.ctx.translate(-originX, -originY);        
    }
  };


  var detection = new FaceDetect(document.getElementById("webcam"), document.getElementById("canvas"));

  detection.addPropType("mask", 0.5, 0.5, "height");

  detection.addProp("/static/SVGS/war-face-mask.svg", "mask");

}(this, document));
