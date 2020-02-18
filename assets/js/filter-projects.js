$('body').on('click', '.project-tag', function(){
    var theme = $(this).data('showtag');

    $(".thumb-unit")
        .fadeOut("fast")
        .promise()
        .done(function(){
            $('.thumb-unit.'+theme).fadeIn("fast");
        });
});

$('body').on('click', '.show-all-projects', function(){
    $('.thumb-unit').fadeIn("fast");
});
