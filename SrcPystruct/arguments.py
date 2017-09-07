class Arguments(object):

    @staticmethod
    def logArguments(class_name, arguments_dict):

        import logging
        logger = logging.getLogger('simple_example')
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

        logging.info("Class name: " + str(class_name))      # print class name

        for name, val in arguments_dict.iteritems():        # print arguments
            if name != 'saved_args' and name != 'self':
                logging.info("Argument: " + str(name) + ": " + str(val))