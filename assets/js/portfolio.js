function shuffleProjects() {
	var container = document.getElementById("projects-container");
	var elementsArray = Array.prototype.slice.call(container.getElementsByClassName('thumb-unit'));
	  elementsArray.forEach(function(element){
	  container.removeChild(element);
	})
	shuffleArray(elementsArray);
	elementsArray.forEach(function(element){
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