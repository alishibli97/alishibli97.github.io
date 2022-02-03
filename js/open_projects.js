


function myFunction(x) {
    alert(x+" heyyyyy man");
}

function openthispage(param){
    var coll = document.getElementsByClassName("g_item_text");
    var i;

    alert(coll.length);

    for (i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            alert("content is: "+content);
            if (content.style.display === "block") {
            content.style.display = "none";
            } else {
            content.style.display = "block";
            }
        });
    }
}