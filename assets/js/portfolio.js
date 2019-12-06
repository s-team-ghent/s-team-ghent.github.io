function shuffleProjects() {
	var container = document.getElementById("projects-container");
	var currElementsArray = Array.prototype.slice.call(container.getElementsByClassName('current-thumb-unit'));
	currElementsArray.forEach(function(element){
	  container.removeChild(element);
	})
	var oldElementsArray = Array.prototype.slice.call(container.getElementsByClassName('old-thumb-unit'));
	oldElementsArray.forEach(function(element){
	container.removeChild(element);
  	})
	currElementsArray = shuffleArray(currElementsArray);
	currElementsArray.forEach(function(element){
		container.appendChild(element);
	})
	oldElementsArray = shuffleArray(oldElementsArray);
	oldElementsArray.forEach(function(element){
		container.appendChild(element);
	})
  }
  
  function shuffleArray(array) {
	  for (var i = array.length - 1; i > 0; i--) {
		  var j = Math.floor(Math.random() * (i + 1));
		  var temp = array[i];
		  array[i] = array[j];
		  array[j] = temp;
	  }
	  return array;
  }

  document.addEventListener('DOMContentLoaded', function() {
    shuffleProjects();
}, false);